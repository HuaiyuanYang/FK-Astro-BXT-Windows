// ----------------------------------------------------------------------------
// engine_test.cpp —— InferenceEngine 的独立运行时验证(不依赖 PixInsight)
//
// 它做的事:
//    1. 通过 BXT_ORT_DLL 指向 PI 自带的 onnxruntime.dll(或任意 ORT)
//    2. Load() 一个 .onnx, 打印运行时版本、可用 EP、识别出的输入/输出形状
//    3. 读一张 CHW float32 输入, 跑一次 Predict(), 把 CHW float32 结果写出
//
// 与 py/engine_equiv_test.py 配对使用: 后者用 numpy 走同一条路径
// (CHW -> NHWC -> ORT -> NHWC -> CHW), 两边输出应**逐字节相同**。
// 逐字节相同即证明:
//    * 动态加载 ORT 成功且版本协商正确
//    * NHWC/NCHW 判定正确
//    * 输入 CHW->NHWC 暂存 与 输出 NHWC->CHW 回写 的索引正确
//    * 参数张量布局与补零正确
//
// 用法:
//    engine_test.exe <model.onnx> <in.f32> <out.f32> [param0 param1 ...]
// ----------------------------------------------------------------------------

#include "InferenceEngine.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>

static bool ReadRaw( const char* path, std::vector<float>& buf, size_t expect )
{
   std::ifstream f( path, std::ios::binary );
   if ( !f )
   {
      std::printf( "CANNOT-OPEN %s\n", path );
      return false;
   }
   buf.resize( expect );
   f.read( reinterpret_cast<char*>( buf.data() ),
           std::streamsize( expect * sizeof( float ) ) );
   if ( size_t( f.gcount() ) != expect * sizeof( float ) )
   {
      std::printf( "SHORT-READ %s: got %lld floats, expected %zu\n",
                   path, (long long)( f.gcount() / sizeof( float ) ), expect );
      return false;
   }
   return true;
}

static bool WriteRaw( const char* path, const std::vector<float>& buf )
{
   std::ofstream f( path, std::ios::binary );
   if ( !f )
      return false;
   f.write( reinterpret_cast<const char*>( buf.data() ),
            std::streamsize( buf.size() * sizeof( float ) ) );
   return bool( f );
}

int main( int argc, char** argv )
{
   if ( argc < 4 )
   {
      std::printf( "usage: engine_test <model.onnx> <in.f32> <out.f32> [params...]\n" );
      return 2;
   }

   fka::InferenceEngine& eng = fka::InferenceEngine::Get();

   if ( !eng.Load( argv[1], fka::DeviceAuto ) )
   {
      std::printf( "LOAD-FAILED\n%s\n", eng.LastError().c_str() );
      return 1;
   }

   std::printf( "RUNTIME   %s\n", eng.RuntimeVersion().c_str() );
   for ( size_t i = 0; i < eng.AvailableProviders().size(); ++i )
      std::printf( "EP        %s\n", eng.AvailableProviders()[i].c_str() );

   const fka::TensorShape in  = eng.InputShape();
   const fka::TensorShape out = eng.OutputShape();

   std::printf( "INPUT     c=%d h=%d w=%d  (%zu floats)\n",
                in.c, in.h, in.w, in.Size() );
   std::printf( "OUTPUT    c=%d h=%d w=%d  (%zu floats)\n",
                out.c, out.h, out.w, out.Size() );
   std::printf( "PARAMS    %zu\n", eng.ParamsSize() );
   std::printf( "DEVICE    %d\n", eng.EffectiveDevice() );

   if ( !in.IsValid() || !out.IsValid() )
   {
      std::printf( "BAD-SHAPES\n" );
      return 1;
   }

   std::vector<float> ibuf;
   if ( !ReadRaw( argv[2], ibuf, in.Size() ) )
      return 1;

   std::vector<float> obuf( out.Size(), 0.0f );

   std::vector<float> params;
   for ( int i = 4; i < argc; ++i )
      params.push_back( float( std::atof( argv[i] ) ) );

   if ( !eng.Predict( ibuf.data(), obuf.data(),
                      params.empty() ? nullptr : params.data(),
                      params.size() ) )
   {
      std::printf( "PREDICT-FAILED\n%s\n", eng.LastError().c_str() );
      return 1;
   }

   if ( !WriteRaw( argv[3], obuf ) )
   {
      std::printf( "CANNOT-WRITE %s\n", argv[3] );
      return 1;
   }

   // 输出统计, 便于两边对照
   double sum = 0.0;
   float lo = obuf[0], hi = obuf[0];
   for ( size_t i = 0; i < obuf.size(); ++i )
   {
      sum += obuf[i];
      if ( obuf[i] < lo ) lo = obuf[i];
      if ( obuf[i] > hi ) hi = obuf[i];
   }
   std::printf( "OUT-STAT  min=%.9g max=%.9g mean=%.9g\n",
                lo, hi, sum / double( obuf.size() ) );
   std::printf( "OK %zu floats written to %s\n", obuf.size(), argv[3] );
   return 0;
}
