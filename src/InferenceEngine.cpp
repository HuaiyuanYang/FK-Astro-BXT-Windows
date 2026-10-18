// ----------------------------------------------------------------------------
// InferenceEngine.cpp —— ONNX Runtime 推理引擎实现(Windows)
//
// 设计要点见 InferenceEngine.h。这里只强调三处容易写错的地方:
//
//   1) ORT 1.25 的 CreateSession 参数顺序是 (env, model_path, options, out),
//      **model_path 在 options 之前** —— 老版本是反的, 照老写法编会静默错。
//   2) 所有 Release* 在头文件里是 ORT_CLASS_RELEASE(Env) 之类的宏展开,
//      不是字面的 ORT_API2_STATUS(ReleaseEnv,...) —— 搜不到不等于不存在。
//   3) 输出张量交给 ORT 分配(outputs[i] 传 nullptr), 我们只负责把结果
//      从模型布局转成调用方要的 CHW。这样不必预判 ORT 的内部布局。
// ----------------------------------------------------------------------------

#include "InferenceEngine.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include "onnxruntime_c_api.h"

#include <cstring>
#include <algorithm>

namespace fka
{

namespace
{

// ---- UTF-8 <-> UTF-16 ----
std::wstring ToWide( const std::string& s )
{
   if ( s.empty() )
      return std::wstring();
   int n = ::MultiByteToWideChar( CP_UTF8, 0, s.c_str(), int( s.size() ), nullptr, 0 );
   if ( n <= 0 )
      return std::wstring();
   std::wstring w( size_t( n ), L'\0' );
   ::MultiByteToWideChar( CP_UTF8, 0, s.c_str(), int( s.size() ), &w[0], n );
   return w;
}

std::string ToUtf8( const std::wstring& w )
{
   if ( w.empty() )
      return std::string();
   int n = ::WideCharToMultiByte( CP_UTF8, 0, w.c_str(), int( w.size() ), nullptr, 0, nullptr, nullptr );
   if ( n <= 0 )
      return std::string();
   std::string s( size_t( n ), '\0' );
   ::WideCharToMultiByte( CP_UTF8, 0, w.c_str(), int( w.size() ), &s[0], n, nullptr, nullptr );
   return s;
}

// 本模块自身所在目录(即 PI 的 bin\)
std::wstring OwnModuleDir()
{
   HMODULE hm = nullptr;
   if ( !::GetModuleHandleExW( GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                               | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCWSTR>( &OwnModuleDir ), &hm ) )
      return std::wstring();

   wchar_t buf[ MAX_PATH ] = { 0 };
   DWORD n = ::GetModuleFileNameW( hm, buf, MAX_PATH );
   if ( n == 0 )
      return std::wstring();

   std::wstring path( buf, n );
   size_t pos = path.find_last_of( L"\\/" );
   return ( pos == std::wstring::npos ) ? std::wstring() : path.substr( 0, pos );
}

// onnxruntime.dll 的完整路径: 环境变量优先, 否则取本模块目录
std::wstring OrtDllPath()
{
   wchar_t env[ 1024 ] = { 0 };
   DWORD n = ::GetEnvironmentVariableW( L"BXT_ORT_DLL", env, 1024 );
   if ( n > 0 && n < 1024 )
      return std::wstring( env, n );

   std::wstring dir = OwnModuleDir();
   if ( dir.empty() )
      return L"onnxruntime.dll";
   return dir + L"\\onnxruntime.dll";
}

} // anonymous namespace

// ----------------------------------------------------------------------------

InferenceEngine& InferenceEngine::Get()
{
   static InferenceEngine s_instance;
   return s_instance;
}

InferenceEngine::~InferenceEngine()
{
   Unload();
}

std::string InferenceEngine::TakeError( void* status )
{
   OrtStatus* st = static_cast<OrtStatus*>( status );
   if ( st == nullptr )
      return std::string();

   const OrtApi* api = static_cast<const OrtApi*>( m_ortApi );
   std::string msg = "unknown ONNX Runtime error";
   if ( api != nullptr )
   {
      const char* m = api->GetErrorMessage( st );
      if ( m != nullptr )
         msg = m;
      api->ReleaseStatus( st );
   }
   return msg;
}

bool InferenceEngine::EnsureRuntime()
{
   if ( m_ortApi != nullptr )
      return true;

   std::wstring dll = OrtDllPath();
   HMODULE h = ::LoadLibraryW( dll.c_str() );
   if ( h == nullptr )
   {
      m_lastError = "Cannot load onnxruntime.dll at \"" + ToUtf8( dll )
                    + "\" (GetLastError=" + std::to_string( ::GetLastError() ) + "). "
                      "Set BXT_ORT_DLL to override.";
      return false;
   }

   typedef const OrtApiBase* ( ORT_API_CALL *GetApiBaseFn )( void );
   GetApiBaseFn getBase =
      reinterpret_cast<GetApiBaseFn>( ::GetProcAddress( h, "OrtGetApiBase" ) );
   if ( getBase == nullptr )
   {
      ::FreeLibrary( h );
      m_lastError = "\"" + ToUtf8( dll ) + "\" does not export OrtGetApiBase; "
                    "it is probably not an ONNX Runtime library.";
      return false;
   }

   const OrtApiBase* base = getBase();
   const OrtApi* api = ( base != nullptr ) ? base->GetApi( ORT_API_VERSION ) : nullptr;
   if ( api == nullptr )
   {
      ::FreeLibrary( h );
      m_lastError = "ONNX Runtime version negotiation failed: the library does not "
                    "provide ORT_API_VERSION " + std::to_string( ORT_API_VERSION ) + ".";
      return false;
   }

   m_ortModule = h;
   m_ortApi = const_cast<OrtApi*>( api );

   const char* vs = base->GetVersionString();
   m_runtimeVersion = ( vs != nullptr ) ? vs : "unknown";

   char** prov = nullptr;
   int count = 0;
   if ( api->GetAvailableProviders( &prov, &count ) == nullptr )
   {
      for ( int i = 0; i < count; ++i )
         if ( prov[i] != nullptr )
            m_providers.push_back( prov[i] );
      api->ReleaseAvailableProviders( prov, count );
   }
   return true;
}

void InferenceEngine::ReleaseAll()
{
   const OrtApi* api = static_cast<const OrtApi*>( m_ortApi );

   if ( api != nullptr )
   {
      if ( m_session != nullptr )
         api->ReleaseSession( static_cast<OrtSession*>( m_session ) );
      if ( m_memoryInfo != nullptr )
         api->ReleaseMemoryInfo( static_cast<OrtMemoryInfo*>( m_memoryInfo ) );
      if ( m_env != nullptr )
         api->ReleaseEnv( static_cast<OrtEnv*>( m_env ) );
   }

   m_session = nullptr;
   m_memoryInfo = nullptr;
   m_env = nullptr;

   if ( m_ortModule != nullptr )
   {
      ::FreeLibrary( static_cast<HMODULE>( m_ortModule ) );
      m_ortModule = nullptr;
   }
   m_ortApi = nullptr;
}

void InferenceEngine::Unload()
{
   ReleaseAll();
   m_loaded = false;
   m_loadedPath.clear();
   m_inputName.clear();
   m_outputName.clear();
   m_paramsName.clear();
   m_inputShape = TensorShape();
   m_outputShape = TensorShape();
   m_paramsSize = 0;
   m_inStage.clear();
   m_paramsStage.clear();
   m_providers.clear();
   m_runtimeVersion.clear();
}

bool InferenceEngine::Load( const std::string& path, int device )
{
   if ( m_loaded && m_loadedPath == path && m_device == device )
      return true;

   Unload();
   m_device = device;
   m_effectiveDevice = device;

   if ( !EnsureRuntime() )
      return false;

   const OrtApi* api = static_cast<const OrtApi*>( m_ortApi );
   OrtStatus* st = nullptr;

   OrtEnv* env = nullptr;
   if ( ( st = api->CreateEnv( ORT_LOGGING_LEVEL_WARNING, "bxt-inference", &env ) ) != nullptr )
   {
      m_lastError = TakeError( st );
      return false;
   }
   m_env = env;
   api->DisableTelemetryEvents( env );

   OrtSessionOptions* so = nullptr;
   if ( ( st = api->CreateSessionOptions( &so ) ) != nullptr )
   {
      m_lastError = TakeError( st );
      ReleaseAll();
      return false;
   }
   api->SetIntraOpNumThreads( so, 0 );          // 0 = 让 ORT 自行决定
   api->SetInterOpNumThreads( so, 1 );
   api->SetSessionGraphOptimizationLevel( so, ORT_ENABLE_ALL );
   api->SetSessionLogSeverityLevel( so, 3 );    // 3 = ERROR

   // ⚠️ ORT 1.25: (env, model_path, options, out)
   std::wstring wpath = ToWide( path );
   OrtSession* sess = nullptr;
   st = api->CreateSession( env, wpath.c_str(), so, &sess );
   api->ReleaseSessionOptions( so );
   if ( st != nullptr )
   {
      m_lastError = TakeError( st );
      ReleaseAll();
      return false;
   }
   m_session = sess;

   OrtMemoryInfo* mi = nullptr;
   if ( ( st = api->CreateCpuMemoryInfo( OrtArenaAllocator, OrtMemTypeDefault, &mi ) ) != nullptr )
   {
      m_lastError = TakeError( st );
      ReleaseAll();
      return false;
   }
   m_memoryInfo = mi;

   // ---- 发现输入/输出 ----
   OrtAllocator* alloc = nullptr;
   if ( api->GetAllocatorWithDefaultOptions( &alloc ) != nullptr || alloc == nullptr )
   {
      m_lastError = "Cannot obtain the default ORT allocator.";
      ReleaseAll();
      return false;
   }

   size_t nInputs = 0;
   api->SessionGetInputCount( sess, &nInputs );

   for ( size_t i = 0; i < nInputs; ++i )
   {
      char* nm = nullptr;
      if ( api->SessionGetInputName( sess, i, alloc, &nm ) != nullptr || nm == nullptr )
         continue;
      std::string name = nm;
      alloc->Free( alloc, nm );

      OrtTypeInfo* ti = nullptr;
      if ( api->SessionGetInputTypeInfo( sess, i, &ti ) != nullptr || ti == nullptr )
         continue;

      const OrtTensorTypeAndShapeInfo* tsi = nullptr;
      if ( api->CastTypeInfoToTensorInfo( ti, &tsi ) != nullptr || tsi == nullptr )
      {
         api->ReleaseTypeInfo( ti );
         continue;
      }

      size_t rank = 0;
      api->GetDimensionsCount( tsi, &rank );
      std::vector<int64_t> dims( rank, 0 );
      if ( rank > 0 )
         api->GetDimensions( tsi, dims.data(), rank );

      size_t elems = 0;
      api->GetTensorShapeElementCount( tsi, &elems );

      api->ReleaseTypeInfo( ti );

      // 判定: 2 维 = 参数张量; 其余第一个 = 图像
      if ( rank == 2 && m_paramsName.empty() )
      {
         m_paramsName = name;
         m_paramsSize = elems;
         m_paramsDims.assign( dims.begin(), dims.end() );
      }
      else if ( m_inputName.empty() )
      {
         m_inputName = name;
         m_inputRank = int( rank );

         // 4-D 时判断 NHWC / NCHW, 并归一化成 {c,h,w}
         if ( rank == 4 )
         {
            const int64_t d1 = dims[1], d2 = dims[2], d3 = dims[3];
            if ( d3 <= 4 && d1 > 4 )        // [N,H,W,C]
            {
               m_inputNHWC = true;
               m_inputShape.c = int( d3 );
               m_inputShape.h = int( d1 );
               m_inputShape.w = int( d2 );
            }
            else if ( d1 <= 4 && d3 > 4 )   // [N,C,H,W]
            {
               m_inputNHWC = false;
               m_inputShape.c = int( d1 );
               m_inputShape.h = int( d2 );
               m_inputShape.w = int( d3 );
            }
            else                            // 分不清时按 NHWC, 并在最后报出来
            {
               m_inputNHWC = true;
               m_inputShape.c = int( d3 );
               m_inputShape.h = int( d1 );
               m_inputShape.w = int( d2 );
            }
         }
         else if ( rank == 3 )
         {
            m_inputNHWC = false;
            m_inputShape.c = int( dims[0] );
            m_inputShape.h = int( dims[1] );
            m_inputShape.w = int( dims[2] );
         }
         m_inputDims.assign( dims.begin(), dims.end() );
      }
   }

   size_t nOutputs = 0;
   api->SessionGetOutputCount( sess, &nOutputs );
   if ( nOutputs == 0 )
   {
      m_lastError = "The model has no outputs.";
      ReleaseAll();
      return false;
   }

   {
      char* nm = nullptr;
      if ( api->SessionGetOutputName( sess, 0, alloc, &nm ) == nullptr && nm != nullptr )
      {
         m_outputName = nm;
         alloc->Free( alloc, nm );
      }

      OrtTypeInfo* ti = nullptr;
      if ( api->SessionGetOutputTypeInfo( sess, 0, &ti ) == nullptr && ti != nullptr )
      {
         const OrtTensorTypeAndShapeInfo* tsi = nullptr;
         if ( api->CastTypeInfoToTensorInfo( ti, &tsi ) == nullptr && tsi != nullptr )
         {
            size_t rank = 0;
            api->GetDimensionsCount( tsi, &rank );
            std::vector<int64_t> dims( rank, 0 );
            if ( rank > 0 )
               api->GetDimensions( tsi, dims.data(), rank );

            if ( rank == 4 )
            {
               const int64_t d1 = dims[1], d2 = dims[2], d3 = dims[3];
               if ( d1 <= 4 && d3 > 4 )     // [N,C,H,W]
               {
                  m_outputNHWC = false;
                  m_outputShape.c = int( d1 );
                  m_outputShape.h = int( d2 );
                  m_outputShape.w = int( d3 );
               }
               else                          // [N,H,W,C]
               {
                  m_outputNHWC = true;
                  m_outputShape.c = int( d3 );
                  m_outputShape.h = int( d1 );
                  m_outputShape.w = int( d2 );
               }
            }
            else if ( rank == 3 )
            {
               m_outputNHWC = false;
               m_outputShape.c = int( dims[0] );
               m_outputShape.h = int( dims[1] );
               m_outputShape.w = int( dims[2] );
            }
            m_outputDims.assign( dims.begin(), dims.end() );
         }
         api->ReleaseTypeInfo( ti );
      }
   }

   if ( m_inputName.empty() || !m_inputShape.IsValid() || !m_outputShape.IsValid()
        || m_outputName.empty() )
   {
      m_lastError = "The model must expose usable tensor input/output ports.";
      ReleaseAll();
      return false;
   }

   m_loaded = true;
   m_loadedPath = path;
   return true;
}

bool InferenceEngine::Predict( const float* in, float* out,
                               const float* params, size_t paramCount )
{
   if ( !m_loaded || in == nullptr || out == nullptr )
   {
      m_lastError = "Predict() called on an unloaded engine or with null buffers.";
      return false;
   }

   const OrtApi* api = static_cast<const OrtApi*>( m_ortApi );
   const int C = m_inputShape.c, H = m_inputShape.h, W = m_inputShape.w;
   const int OC = m_outputShape.c, OH = m_outputShape.h, OW = m_outputShape.w;

   // ---- 输入: 需要时 CHW -> NHWC ----
   const void* inData = in;
   if ( m_inputNHWC )
   {
      m_inStage.resize( size_t( C ) * H * W );
      for ( int c = 0; c < C; ++c )
         for ( int y = 0; y < H; ++y )
         {
            const float* src = in + ( size_t( c ) * H + y ) * W;
            float* dst = m_inStage.data() + ( size_t( y ) * W ) * C + c;
            for ( int x = 0; x < W; ++x )
               dst[ size_t( x ) * C ] = src[ x ];
         }
      inData = m_inStage.data();
   }

   OrtValue* inVal = nullptr;
   OrtStatus* st = api->CreateTensorWithDataAsOrtValue(
         static_cast<OrtMemoryInfo*>( m_memoryInfo ),
         const_cast<void*>( inData ),
         size_t( C ) * H * W * sizeof( float ),
         m_inputDims.data(), m_inputDims.size(),
         ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inVal );
   if ( st != nullptr )
   {
      m_lastError = TakeError( st );
      return false;
   }

   const char* inNames[ 2 ] = { m_inputName.c_str(), m_paramsName.c_str() };
   const OrtValue* inVals[ 2 ] = { inVal, nullptr };
   size_t nInputs = 1;

   OrtValue* parVal = nullptr;
   if ( !m_paramsName.empty() )
   {
      m_paramsStage.assign( m_paramsSize, 0.0f );
      if ( params != nullptr && paramCount > 0 )
      {
         const size_t n = ( paramCount < m_paramsSize ) ? paramCount : m_paramsSize;
         std::memcpy( m_paramsStage.data(), params, n * sizeof( float ) );
      }
      st = api->CreateTensorWithDataAsOrtValue(
            static_cast<OrtMemoryInfo*>( m_memoryInfo ),
            m_paramsStage.data(),
            m_paramsSize * sizeof( float ),
            m_paramsDims.data(), m_paramsDims.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &parVal );
      if ( st != nullptr )
      {
         m_lastError = TakeError( st );
         api->ReleaseValue( inVal );
         return false;
      }
      inVals[ 1 ] = parVal;
      nInputs = 2;
   }

   // ---- 跑 ----
   const char* outNames[ 1 ] = { m_outputName.c_str() };
   OrtValue* outVal = nullptr;
   st = api->Run( static_cast<OrtSession*>( m_session ), nullptr,
                  inNames, inVals, nInputs,
                  outNames, 1, &outVal );

   api->ReleaseValue( inVal );
   if ( parVal != nullptr )
      api->ReleaseValue( parVal );

   if ( st != nullptr )
   {
      m_lastError = TakeError( st );
      return false;
   }

   // ---- 输出: 需要时 NHWC -> CHW ----
   float* od = nullptr;
   if ( api->GetTensorMutableData( outVal, reinterpret_cast<void**>( &od ) ) != nullptr
        || od == nullptr )
   {
      m_lastError = "Cannot access the output tensor data.";
      api->ReleaseValue( outVal );
      return false;
   }

   if ( m_outputNHWC )
   {
      for ( int c = 0; c < OC; ++c )
         for ( int y = 0; y < OH; ++y )
         {
            const float* src = od + ( size_t( y ) * OW ) * OC + c;
            float* dst = out + ( size_t( c ) * OH + y ) * OW;
            for ( int x = 0; x < OW; ++x )
               dst[ x ] = src[ size_t( x ) * OC ];
         }
   }
   else
   {
      std::memcpy( out, od, size_t( OC ) * OH * OW * sizeof( float ) );
   }

   api->ReleaseValue( outVal );
   return true;
}

} // namespace fka
