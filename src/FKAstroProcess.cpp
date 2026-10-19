/*
 * fk-astro — FKAstroInference process implementation
 *
 * Runs a user-supplied CoreML model over the pixels of the current view,
 * using tiled inference with overlap margins and linear-ramp blending.
 */

#include <pcl/Console.h>
#include <pcl/Exception.h>
#include <pcl/View.h>
#include <pcl/StatusMonitor.h>

#include <pcl/Thread.h>
#include <pcl/MetaModule.h>
#include <pcl/ReferenceArray.h>
#include <pcl/AbstractImage.h>
#include <pcl/StandardStatus.h>
#include <pcl/AutoStatusCallbackRestorer.h>
#include <pcl/ImageWindow.h>
#include <pcl/ImageOptions.h>
#include <pcl/AutoViewLock.h>
#include <pcl/Image.h>
#include <pcl/ImageVariant.h>
#include <pcl/Utility.h>

#include "FKAstroProcess.h"
#include "InferenceEngine.h"

#include <vector>
#include <algorithm>
#include <type_traits>
#include <chrono>

#include "FKv3Pipeline.h"

namespace
{

// ------------------------------------------------------------------
// Sample -> float conversion (PixInsight float images are [0,1])
// ------------------------------------------------------------------

template <typename S>
float ToNormFloat( S v )
{
   if constexpr ( std::is_floating_point<S>::value )
      return float( v );
   else
      return float( v ) / float( std::numeric_limits<S>::max() );
}

// ---------------------------------------------------------------------------
// 归一化
//
// 现在只有 FKv3: 整图 arcsinh。旧的 FKv1(log + MTF + 仿射, 每块一次)已完全
// 移除 —— 它的常数来自上游 2026-10-13 那一版, 后来被作者替换掉了, 照它算出来
// 的结果与官方相差 4~9 倍, 会把星点推到饱和。
//
// 数学全部在 FKv3Pipeline.h 里, 与离线自检(fkv3_selftest.cpp)共用同一份实现。
// ---------------------------------------------------------------------------
struct FKNormalizer
{
   enum Scheme { None = 0, FKv3 = 1 };

   Scheme scheme = FKv3;
   fka::fkv3::ArcsinhNormalizer v3;

   bool IsV3() const { return scheme == FKv3; }

   float Forward( float x ) const
   {
      return IsV3() ? float( v3.Forward( double( x ) ) ) : x;
   }

   float Inverse( float y ) const
   {
      return IsV3() ? float( v3.Inverse( double( y ) ) ) : y;
   }

   double Gain() const { return IsV3() ? v3.gain : 0.0; }

   double InverseD( double y ) const { return IsV3() ? v3.Inverse( y ) : y; }
};

// ---------------------------------------------------------------------------
// 整图拟合 FKv3 归一化(按样本类型分派)
// ---------------------------------------------------------------------------
template <class P>
static void FKFitV3T( const pcl::GenericImage<P>& img, fka::fkv3::ArcsinhNormalizer& nz,
                      bool lunar )
{
   using sample = typename P::sample;
   const int W = img.Width(), H = img.Height(), N = img.NumberOfChannels();
   nz.Fit( W, H, N, lunar,
           [&]( int c, int y, int x ) -> double
           {
              const sample* p = img.PixelData( c );
              return fka::fkv3::SampleToDouble( p[ size_t( y ) * size_t( W ) + size_t( x ) ] );
           } );
}

static void FKFitV3( const pcl::ImageVariant& image, fka::fkv3::ArcsinhNormalizer& nz,
                     bool lunar )
{
   if ( image.IsFloatSample() )
      switch ( image.BitsPerSample() )
      {
      case 32: FKFitV3T( static_cast<const pcl::Image&>( *image ), nz, lunar ); return;
      case 64: FKFitV3T( static_cast<const pcl::DImage&>( *image ), nz, lunar ); return;
      }
   else
      switch ( image.BitsPerSample() )
      {
      case 8:  FKFitV3T( static_cast<const pcl::UInt8Image&>( *image ), nz, lunar ); return;
      case 16: FKFitV3T( static_cast<const pcl::UInt16Image&>( *image ), nz, lunar ); return;
      case 32: FKFitV3T( static_cast<const pcl::UInt32Image&>( *image ), nz, lunar ); return;
      }
   throw pcl::Error( "fk-astro: unsupported sample format for normalization." );
}

// Local half-float conversions for memory-lean accumulators.
static uint16_t FK_ToHalf( float f );
static float FK_ToFloat( uint16_t h );

static inline uint16_t FK_ToHalfImpl( float f )
{
   uint32_t x;
   std::memcpy( &x, &f, 4 );
   uint32_t sign = ( x >> 16 ) & 0x8000u;
   int32_t  exp  = int32_t( ( x >> 23 ) & 0xFF ) - 127 + 15;
   uint32_t man  = x & 0x7FFFFF;
   if ( int32_t( ( x >> 23 ) & 0xFF ) == 0xFF )
      return uint16_t( sign | 0x7C00u | ( man ? 1u : 0u ) );
   if ( exp >= 0x1F )
      return uint16_t( sign | 0x7C00u );
   if ( exp <= 0 )
      return uint16_t( sign );
   return uint16_t( sign | ( uint32_t( exp ) << 10 ) | ( man >> 13 ) );
}

static inline float FK_ToFloatImpl( uint16_t h )
{
   uint32_t sign = uint32_t( h & 0x8000u ) << 16;
   uint32_t exp  = ( h >> 10 ) & 0x1Fu;
   uint32_t man  = h & 0x3FFu;
   uint32_t bits;
   if ( exp == 0 )
   {
      if ( man == 0 )
         bits = sign;
      else
      {
         exp = 127 - 15 + 1;
         while ( ( man & 0x400u ) == 0 ) { man <<= 1; --exp; }
         man &= 0x3FFu;
         bits = sign | ( exp << 23 ) | ( man << 13 );
      }
   }
   else if ( exp == 0x1F )
      bits = sign | 0x7F800000u | ( man ? 1u : 0u );
   else
      bits = sign | ( ( exp - 15 + 127 ) << 23 ) | ( man << 13 );
   float f;
   std::memcpy( &f, &bits, 4 );
   return f;
}

static uint16_t FK_ToHalf( float f ) { return FK_ToHalfImpl( f ); }
static float FK_ToFloat( uint16_t h ) { return FK_ToFloatImpl( h ); }


// ------------------------------------------------------------------
// Tile helpers. Model buffers use CHW layout: buf[c*H*W + y*W + x].
// ------------------------------------------------------------------

/*
 * Type-erased per-tile image reader: reads the input window directly
 * from the view's image with edge-replicate clamping, converting to
 * normalized float32 on the fly. This avoids a full-image float copy
 * (~300 MB for a 24 MP color image) that caused heavy paging.
 */
struct FKImageReader
{
   virtual ~FKImageReader() = default;
   virtual void ReadTile( float* dst, int pass, int ix, int iy,
                          const fka::TensorShape& s,
                          bool monoModel, bool replicate ) = 0;
};

template <class P>
struct FKImageReaderT : public FKImageReader
{
   pcl::GenericImage<P>& img;
   int W, H, N;
   bool isCFA = false;
   const int (*cfaMap)[2] = nullptr;   // [y&1][x&1] -> channel (0=R,1=G,2=B)
   const FKNormalizer* normalizer = nullptr;
   bool reflectPad = true;

   FKImageReaderT( pcl::GenericImage<P>& img_ ) : img( img_ )
   {
      W = img.Width();
      H = img.Height();
      N = img.NumberOfChannels();
   }

   void ReadTile( float* dst, int pass, int ix, int iy,
                  const fka::TensorShape& s,
                  bool monoModel, bool replicate ) override
   {
      using sample = typename P::sample;
      const int C = s.c, TH = s.h, TW = s.w;

      if ( isCFA )
      {
         // Bilinear demosaic of the window from the mono Bayer plane.
         const sample* plane = img.PixelData( 0 );
         auto at = [&]( int x, int y ) -> float
         {
            if ( reflectPad )
            {
               // mirror without edge repetition: -1 -> 1, W -> W-2
               if ( x < 0 ) x = -x;
               if ( x >= W ) x = 2*W - 2 - x;
               if ( y < 0 ) y = -y;
               if ( y >= H ) y = 2*H - 2 - y;
               x = std::min( std::max( x, 0 ), W - 1 );
               y = std::min( std::max( y, 0 ), H - 1 );
            }
            else
            {
               x = std::min( std::max( x, 0 ), W - 1 );
               y = std::min( std::max( y, 0 ), H - 1 );
            }
            float v = ToNormFloat( plane[ size_t( y )*W + x ] );
            return normalizer->Forward( v );
         };

         for ( int y = 0; y < TH; ++y )
         {
            const int ay = iy + y;
            const int cy = std::min( std::max( ay, 0 ), H - 1 );
            for ( int x = 0; x < TW; ++x )
            {
               const int ax = ix + x;
               const int cx = std::min( std::max( ax, 0 ), W - 1 );
               const int sx = ax & 1, sy = ay & 1;
               const int site = cfaMap[sy][sx];

               float rgb[3] = { 0, 0, 0 };   // already normalized by at()
               if ( site == 1 ) // green site
               {
                  rgb[1] = at( cx, cy );
                  // non-green color sharing this row -> horizontal neighbors
                  const int hc = (cfaMap[sy][0] != 1) ? cfaMap[sy][0] : cfaMap[sy][1];
                  // non-green color sharing this column -> vertical neighbors
                  const int vc = (cfaMap[0][sx] != 1) ? cfaMap[0][sx] : cfaMap[1][sx];
                  rgb[hc] = (at( cx - 1, cy ) + at( cx + 1, cy )) * 0.5f;
                  rgb[vc] = (at( cx, cy - 1 ) + at( cx, cy + 1 )) * 0.5f;
               }
               else
               {
                  rgb[site] = at( cx, cy );
                  rgb[1] = (at( cx - 1, cy ) + at( cx + 1, cy )
                           + at( cx, cy - 1 ) + at( cx, cy + 1 )) * 0.25f;
                  const int diag = 2 - site; // the opposite non-green color
                  rgb[diag] = (at( cx - 1, cy - 1 ) + at( cx + 1, cy - 1 )
                              + at( cx - 1, cy + 1 ) + at( cx + 1, cy + 1 )) * 0.25f;
               }

               for ( int c = 0; c < 3 && c < C; ++c )
                  dst[ size_t( c )*TH*TW + size_t( y )*TW + x ] = rgb[c];
            }
         }
         return;
      }

      for ( int c = 0; c < C; ++c )
      {
         const int srcChannel = monoModel ? pass
                              : (replicate ? 0 : c);
         if ( srcChannel >= N )
            continue;
         const sample* plane = img.PixelData( srcChannel );
         for ( int y = 0; y < TH; ++y )
         {
            const int sy = iy + y;
            float* row = dst + size_t( c )*TH*TW + size_t( y )*TW;
            if ( sy < 0 || sy >= H )
            {
               // 整行在图外: 模型域 0(不是 Forward(0), 也不是镜像)
               std::fill( row, row + TW, 0.0f );
               continue;
            }
            const sample* srcRow = plane + size_t( sy )*W;
            for ( int x = 0; x < TW; ++x )
            {
               const int sx = ix + x;
               row[x] = ( sx < 0 || sx >= W )
                        ? 0.0f
                        : normalizer->Forward( ToNormFloat( srcRow[sx] ) );
            }
         }
      }
   }
};

// ------------------------------------------------------------------
// Type-dispatched pipeline
// ------------------------------------------------------------------

struct RunConfig
{
   fka::CoreMLEngine* engine  = nullptr;
   int                W = 0, H = 0, N = 0;
   fka::TensorShape   inS, outS;
   bool               monoModel = false;  // 模型输入只有 1 个通道
   bool               replicate = false;  // 模型要 C>1, 图像是单色
   int                ctx = 0;            // 输入块比输出块每侧多出的像素
   // CFA(未去马赛克的 Bayer 输入)
   bool               isCFA = false;
   int                cfaMap[2][2] = { { 0, 1 }, { 1, 2 } };
   // FKv3 分块网格
   fka::fkv3::AxisGrid gx, gy;
   bool               verbose = true;
   FKNormalizer       normalizer;
   std::vector<float> params;             // parameters 张量
};

// ------------------------------------------------------------------
// Worker thread: processes a contiguous range of tiles.
// ------------------------------------------------------------------

struct FKAstroThreadData : public pcl::AbstractImage::ThreadData
{
   FKAstroThreadData( const pcl::AbstractImage& image, pcl::size_type N )
      : pcl::AbstractImage::ThreadData( image, N )
   {
   }

   const RunConfig* C = nullptr;
   FKImageReader* reader = nullptr;
   const FKNormalizer* normalizer = nullptr;
   // 模型域累加器: FKv3 的权重带不归一化, 必须用 double
   std::vector<std::vector<double>>* acc = nullptr;
   int passes = 1;
};

class FKAstroTileThread : public pcl::Thread
{
public:

   using size_type = pcl::size_type;

   FKAstroTileThread( FKAstroThreadData& data, size_type begin, size_type end )
      : m_data( data ), m_begin( begin ), m_end( end )
   {
   }

   void Run() override
   {
      // 诊断: 用一个阶段标记定位异常点。INIT_THREAD_MONITOR() 也必须放在
      // try 里面 —— 之前它在外面, 一旦抛异常就会越过本类的 catch 直接穿出去。
      // 注意用 IsoString(ASCII), 因为 pcl::String::c_str() 是 UTF-16。
      pcl::IsoString stage = "entry";
      m_stage = stage;

      try
      {
         // 注意: 这里**不用** INIT_THREAD_MONITOR()。这个进程只有一个工作
         // 单元, 所以 Run() 是直接在主线程上调用执行的, 线程监视器的宏要求
         // 真正的 pcl::Thread 上下文, 在主线程上是不成立的。
         stage = "cfg"; m_stage = stage;
         const RunConfig& C = *m_data.C;
         const int W = C.W, H = C.H, N = C.N;
         const size_t nx = C.gx.Count(), ny = C.gy.Count();
         const size_t perPass = nx * ny;

         std::vector<float> inBuf( C.inS.Size() );
         std::vector<float> outBuf( C.outS.Size() );
         const int outTH = C.outS.h, outTW = C.outS.w;

         for ( size_type t = m_begin; t < m_end; ++t )
         {
            const bool firstTile = ( t == m_begin && C.verbose );
            const int pass = int( size_t( t ) / perPass );
            const size_t r = size_t( t ) % perPass;
            const size_t j = r / nx, i = r % nx;

            const int x0 = C.gx.origins[i];
            const int y0 = C.gy.origins[j];
            const int ix = x0 - C.ctx;
            const int iy = y0 - C.ctx;

            const auto tt = std::chrono::steady_clock::now();
            const bool timingTile = ( size_type( t - m_begin ) < 6 );

            if ( firstTile )
               pcl::Console().WriteLn( "fk-astro: stage: fill" );

            m_data.reader->ReadTile( inBuf.data(), pass, ix, iy,
                                     C.inS, C.monoModel, C.replicate );

            if ( firstTile )
               pcl::Console().WriteLn( "fk-astro: stage: predict" );

            if ( !C.engine->Predict( inBuf.data(), outBuf.data(),
                                     C.params.data(), C.params.size() ) )
            {
               m_error = ( "fk-astro: prediction failed: "
                           + C.engine->LastError() ).c_str();
               return;
            }

            if ( firstTile )
               pcl::Console().WriteLn( "fk-astro: stage: store" );

            // 每块的余弦带权重按需算(512 个 double, 开销可忽略)
            const std::vector<double> wx = C.gx.Window( i );
            const std::vector<double> wy = C.gy.Window( j );

            fka::fkv3::BlendTile( outBuf.data(), C.outS.c, N, outTW, outTH,
                                  x0, y0, W, H, wx, wy,
                                  [&]( int c, size_t dst, double v )
                                  {
                                     const int ch = C.monoModel ? pass : c;
                                     ( *m_data.acc )[ size_t( ch ) ][ dst ] += v;
                                  } );

            if ( firstTile )
               pcl::Console().WriteLn( "fk-astro: stage: tile 1 done" );

            if ( timingTile )
            {
               double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - tt ).count();
               pcl::Console().WriteLn( pcl::IsoString().Format(
                  "fk-astro: tile %d: %.1f ms", int( t ), ms ).c_str() );
            }

            // 进度 + 保持界面响应(在主线程上跑, 每隔若干块处理一次事件)
            m_data.status += 1;
            if ( ( t & 3 ) == 0 )
               pcl::Module->ProcessEvents( /*excludeUserInputEvents*/ true );
         }
      }
      catch ( const std::exception& e )
      {
         m_error = pcl::String( ( std::string( "fk-astro: thread exception at [" )
                                  + std::string( m_stage.c_str() ) + "]: "
                                  + e.what() ).c_str() );
      }
      catch ( ... )
      {
         m_error = pcl::String( ( std::string( "fk-astro: unknown thread exception at [" )
                                  + std::string( m_stage.c_str() ) + "]" ).c_str() );
      }
   }
   pcl::String m_error;
   pcl::IsoString m_stage = "ctor";

private:

   FKAstroThreadData& m_data;
   size_type          m_begin;
   size_type          m_end;
};

// ------------------------------------------------------------------
// Type-dispatched pipeline
// ------------------------------------------------------------------

template <class P>
void RunPipeline( pcl::GenericImage<P>& img, const RunConfig& C )
{
   using sample = typename P::sample;

   const int W = C.W, H = C.H, N = C.N;
   const size_t plane = size_t( W ) * size_t( H );

   FKImageReaderT<P> reader( img );
   reader.isCFA = C.isCFA;
   reader.cfaMap = C.cfaMap;
   reader.normalizer = &C.normalizer;

   const int passes = C.monoModel ? N : 1;

   // 模型域累加器。FKv3 的余弦带权重不带归一化分母, 必须用 double 累加。
   std::vector<std::vector<double>> acc( size_t( N ), std::vector<double>( plane, 0.0 ) );

   const size_t total = C.gx.Count() * C.gy.Count() * size_t( passes );

   pcl::StandardStatus stdStatus;
   pcl::AutoStatusCallbackRestorer restoreStatusCallback( img.Status() );
   img.SetStatusCallback( &stdStatus );

   FKAstroThreadData data( img, total );
   data.normalizer = &C.normalizer;
   data.status.Initialize( pcl::String( "fk-astro: applying model" ),
                           pcl::size_type( total ) );
   data.C = &C;
   data.reader = &reader;
   data.acc = &acc;
   data.passes = passes;

   if ( C.verbose )
      pcl::Console().WriteLn( pcl::IsoString().Format(
         "fk-astro: stage: begin %d tile(s) [serial]", int( total ) ).c_str() );

   // 串行执行 —— 这个进程只有一个工作单元, pcl::Thread 只是多一层间接。
   // 之前异常在"主线程等待线程"那段被吞掉, 根本定位不到真正出错的位置。
   // 现在直接在主线程调用 Run(): 异常会带着 m_stage 直接抛上来。
   {
      FKAstroTileThread worker( data, 0, total );
      worker.Run();
      if ( !worker.m_error.IsEmpty() )
         throw pcl::Error( worker.m_error );
   }

   if ( C.verbose )
      pcl::Console().WriteLn( "fk-astro: stage: all tiles done" );

   // ---- 写出: 整图只做一次逆变换, 且不除以权重和 ----
   if ( C.isCFA )
   {
      sample* d = img.PixelData( 0 );
      for ( int y = 0; y < H; ++y )
      {
         const int sy = y & 1;
         for ( int x = 0; x < W; ++x )
         {
            const int c = C.cfaMap[sy][x & 1];
            const size_t i = size_t( y ) * size_t( W ) + size_t( x );
            double v = C.normalizer.InverseD( acc[ size_t( c ) ][i] );
            if ( v < 0.0 ) v = 0.0; else if ( v > 1.0 ) v = 1.0;
            d[i] = P::FloatToSample( float( v ) );
         }
      }
   }
   else
   {
      for ( int c = 0; c < N; ++c )
      {
         sample* d = img.PixelData( c );
         const double* s = acc[ size_t( c ) ].data();
         for ( size_t i = 0; i < plane; ++i )
         {
            double v = C.normalizer.InverseD( s[i] );
            if ( v < 0.0 ) v = 0.0; else if ( v > 1.0 ) v = 1.0;
            d[i] = P::FloatToSample( float( v ) );
         }
      }
   }
}


} // anonymous namespace

// ---------------------------------------------------------------------------
// Parameter singletons
// ---------------------------------------------------------------------------

FKAstroInferenceProcess* TheFKAstroInferenceProcess = nullptr;

FKModelPathParameter*    TheFKModelPathParameter    = nullptr;
FKSharpenStarsParameter* TheFKSharpenStarsParameter = nullptr;
FKAdjustStarHalosParameter* TheFKAdjustStarHalosParameter = nullptr;
FKSharpenNonstellarParameter* TheFKSharpenNonstellarParameter = nullptr;
FKAutoPSFParameter* TheFKAutoPSFParameter = nullptr;
FKPSFDiameterParameter* TheFKPSFDiameterParameter = nullptr;
FKCorrectOnlyParameter* TheFKCorrectOnlyParameter = nullptr;
FKNormSchemeParameter* TheFKNormSchemeParameter = nullptr;
FKReflectPadParameter* TheFKReflectPadParameter = nullptr;
FKTileOverlapParameter*  TheFKTileOverlapParameter  = nullptr;
FKDeviceParameter*       TheFKDeviceParameter       = nullptr;
FKVerboseParameter*      TheFKVerboseParameter      = nullptr;

// ---------------------------------------------------------------------------
// FKAstroInferenceProcess
// ---------------------------------------------------------------------------

FKAstroInferenceProcess::FKAstroInferenceProcess()
   : pcl::MetaProcess()
{
   TheFKAstroInferenceProcess = this;
   TheFKModelPathParameter    = nullptr;
   new FKModelPathParameter( this );
   new FKSharpenStarsParameter( this );
   new FKAdjustStarHalosParameter( this );
   new FKSharpenNonstellarParameter( this );
   new FKAutoPSFParameter( this );
   new FKPSFDiameterParameter( this );
   new FKCorrectOnlyParameter( this );
   new FKNormSchemeParameter( this );
   new FKReflectPadParameter( this );
   new FKTileOverlapParameter( this );
   new FKDeviceParameter( this );
   new FKVerboseParameter( this );
}

pcl::IsoString FKAstroInferenceProcess::Id() const
{
   return "FKAstroInference";
}

pcl::IsoString FKAstroInferenceProcess::Aliases() const
{
   return "CoreMLInference,fkAstroInference";
}

pcl::IsoString FKAstroInferenceProcess::Categories() const
{
   return "Preprocessing,FKAstro";
}

pcl::IsoString FKAstroInferenceProcess::IconImageSVG() const
{
   return
      "<svg xmlns='http://www.w3.org/2000/svg' width='128' height='128'>"
      "<defs>"
      "<linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>"
      "<stop offset='0' stop-color='#0b3d5c'/><stop offset='1' stop-color='#17a2b8'/>"
      "</linearGradient>"
      "</defs>"
      "<rect width='128' height='128' rx='18' fill='url(#g)'/>"
      "<circle cx='64' cy='56' r='20' fill='#e9f6fb' opacity='0.9'/>"
      "<ellipse cx='64' cy='102' rx='38' ry='10' fill='#e9f6fb' opacity='0.45'/>"
      "<path d='M28 96 L52 74 M100 96 L76 74' stroke='#ffd166' stroke-width='6' stroke-linecap='round'/>"
      "</svg>";
}

pcl::ProcessImplementation* FKAstroInferenceProcess::Create() const
{
   return new FKAstroInferenceInstance( this );
}

pcl::ProcessImplementation* FKAstroInferenceProcess::Clone( const pcl::ProcessImplementation& p ) const
{
   FKAstroInferenceInstance* instance = new FKAstroInferenceInstance( this );
   instance->Assign( p );
   return instance;
}

// ---------------------------------------------------------------------------
// FKAstroInferenceInstance
// ---------------------------------------------------------------------------

FKAstroInferenceInstance::FKAstroInferenceInstance( const FKAstroInferenceProcess* p )
   : pcl::ProcessImplementation( p )
{
}

void FKAstroInferenceInstance::Assign( const pcl::ProcessImplementation& x )
{
   const FKAstroInferenceInstance& p = dynamic_cast<const FKAstroInferenceInstance&>( x );
   p_modelPath    = p.p_modelPath;
   p_sharpenStars      = p.p_sharpenStars;
   p_adjustStarHalos   = p.p_adjustStarHalos;
   p_sharpenNonstellar = p.p_sharpenNonstellar;
   p_autoPSF      = p.p_autoPSF;
   p_psfDiameter  = p.p_psfDiameter;
   p_tileOverlap  = p.p_tileOverlap;
   p_correctOnly  = p.p_correctOnly;
   p_normScheme   = p.p_normScheme;
   p_reflectPad   = p.p_reflectPad;
   p_device       = p.p_device;
   p_verbose      = p.p_verbose;
}

bool FKAstroInferenceInstance::Validate( pcl::String& info )
{
   if ( p_modelPath.IsEmpty() )
   {
      info = "No CoreML model file has been specified.";
      return false;
   }
   return true;
}

bool FKAstroInferenceInstance::CanExecuteOn( const pcl::View& view, pcl::String& whyNot ) const
{
   if ( !view.IsMainView() )
   {
      whyNot = "fk-astro can only be executed on main views.";
      return false;
   }
   if ( view.Image().IsComplexSample() )
   {
      whyNot = "fk-astro cannot be executed on complex images.";
      return false;
   }
   return true;
}

bool FKAstroInferenceInstance::ExecuteOn( pcl::View& view )
{
   if ( p_modelPath.IsEmpty() )
      throw pcl::Error( "fk-astro: no CoreML model specified." );

   pcl::AutoViewLock lock( view );

   if ( p_verbose )
      pcl::Console().WriteLn( pcl::IsoString().Format(
         "fk-astro: ExecuteOn on %s thread",
         pcl::Thread::IsRootThread() ? "ROOT" : "worker" ).c_str() );

   const auto t0 = std::chrono::steady_clock::now();

   pcl::ImageVariant image = view.Image();
   if ( image.Width() < 1 || image.Height() < 1 )
      throw pcl::Error( "fk-astro: the target view is empty." );
   if ( image.IsComplexSample() )
      throw pcl::Error( "fk-astro cannot be executed on complex images." );

   fka::CoreMLEngine& engine = fka::CoreMLEngine::Get();
   if ( !engine.Load( pcl::IsoString( p_modelPath ).c_str(), int( p_device ) ) )
      throw pcl::Error( ( "fk-astro: cannot load model: "
                          + engine.LastError() ).c_str() );

   const fka::TensorShape inS  = engine.InputShape();
   const fka::TensorShape outS = engine.OutputShape();
   if ( !inS.IsValid() || !outS.IsValid() )
      throw pcl::Error( "fk-astro: the model must expose usable multi-array ports." );
   if ( outS.c != inS.c )
      throw pcl::Error( pcl::IsoString().Format(
         "fk-astro: model has %d input channel(s) but %d output channel(s); "
         "these must match.", inS.c, outS.c ) );

   RunConfig C;
   C.engine = &engine;
   C.W = image.Width();
   C.H = image.Height();
   C.N = image.NumberOfChannels();
   C.inS = inS;
   C.outS = outS;
   C.verbose = p_verbose;
   // reflectPad 在 FKv3 下不再适用: 越界区域固定填模型域 0。

   // ---- normalization scheme (PIPELINE.md) ----
   //   None : pixels pass through, 5 user parameter slots only.
   // ---- 归一化: 整图 arcsinh(FKv3) ----
   //   FKv1 已移除。第 6 个参数槽是 gain = log10(x/a), 由拟合状态直接给出,
   //   不再是旧的 -log10(bg) —— 后者会让模型行为完全跑偏。
   const bool useV3 = ( int( p_normScheme ) == 1 );

   if ( useV3 )
   {
      C.normalizer.scheme = FKNormalizer::FKv3;
      FKFitV3( image, C.normalizer.v3, /*lunar*/ false );

      if ( C.normalizer.v3.positiveSamples == 0 )
         throw pcl::Error( "fk-astro: the image has no positive pixels; "
                           "cannot fit the normalization." );

      if ( p_verbose )
         pcl::Console().WriteLn( pcl::IsoString().Format(
            "fk-astro: FKv3: low=%.6g high=%.6g median=%.6g x=%.6g gain=%.6f "
            "(crop %dx%d of %dx%d, %d samples)",
            C.normalizer.v3.low, C.normalizer.v3.high, C.normalizer.v3.median,
            C.normalizer.v3.x, C.normalizer.v3.gain,
            C.normalizer.v3.cropW, C.normalizer.v3.cropH, C.W, C.H,
            int( C.normalizer.v3.positiveSamples ) ).c_str() );
   }
   else
   {
      C.normalizer.scheme = FKNormalizer::None;
      if ( p_verbose )
         pcl::Console().WriteLn( "fk-astro: normalization disabled (None)" );
   }

   // ---- parameters tensor payload (model contract: [1,6]) ----
   //   [0] sharpenStars        [1] adjustStarHalos
   //   [2] psfDiameter/2       [3] autoPSF 标志
   //   [4] sharpenNonstellar   [5] gain = log10(x/a)
   if ( engine.HasParamsInput() )
   {
      C.params.assign( engine.ParamsSize(), 0.0f );

      // 与 python/fkv3.py 的 parameters() 逐行对齐 —— 那边是与官方实现
      // 对过账的版本, 这里不能自作主张。
      //
      // 顺序(对应 upstream Settings::Effective()):
      //   correctOnly -> stars/halos/diameter/nonstellar 归零, autoPSF = true
      //   lunar       -> stars/halos 归零, autoPSF = false
      //   autoPSF     -> diameter 归零
      //
      // 特别注意第 4 槽: 它是 autoPSF 标志, **correct-only 时是 1, 不是 0**。
      // 这里曾经硬写成 0(依据一段 FKv1 时代的实测注释, 那个结论在 FKv3 下
      // 不成立)。实测在真实数据上, 槽 4 取 0 会让输出与正确结果差到
      // 满量程的 58%(correct-only)/ 76%(sharpen)。
      double stars    = p_correctOnly ? 0.0 : double( p_sharpenStars );
      double halos    = p_correctOnly ? 0.0 : double( p_adjustStarHalos );
      double nonst    = p_correctOnly ? 0.0 : double( p_sharpenNonstellar );
      double diameter = p_correctOnly ? 0.0 : double( p_psfDiameter );
      bool   autoPSF  = p_correctOnly ? true : bool( p_autoPSF );

      if ( autoPSF )
         diameter = 0.0;

      if ( C.params.size() > 0 ) C.params[0] = float( stars );
      if ( C.params.size() > 1 ) C.params[1] = float( halos );
      if ( C.params.size() > 2 ) C.params[2] = float( diameter * 0.5 );
      if ( C.params.size() > 3 ) C.params[3] = autoPSF ? 1.0f : 0.0f;
      if ( C.params.size() > 4 ) C.params[4] = float( nonst );
      // 第 6 槽: gain = log10(x/a), 由归一化拟合状态给出(不是用户参数)。
      if ( useV3 && C.params.size() > 5 )
         C.params[5] = float( C.normalizer.Gain() );
   }

   // ---- verbose: report the exact parameters tensor payload ----
   if ( p_verbose )
   {
      pcl::IsoString pmsg;
      if ( engine.HasParamsInput() )
      {
         pmsg << "fk-astro: params["
              << pcl::IsoString().Format( "%d", int( C.params.size() ) ) << "]:";
         const char* names[6] = { " sharpenStars", " adjustStarHalos",
                                  " psfDiameter/2", " autoPSF",
                                  " sharpenNonstellar", " gain" };
         for ( size_t i = 0; i < C.params.size(); ++i )
         {
            pmsg << ( i < 6 ? names[i] : " unused" )
                 << pcl::IsoString().Format( "=%.4f", double( C.params[i] ) );
         }
         if ( p_correctOnly )
            pmsg << "  (correct only: adjustments neutralized)";
         if ( !useV3 && C.params.size() > 5 )
            pmsg << "  (slot 5 left at 0; select FKv3 to compute it)";
      }
      else
         pmsg = "fk-astro: params: model has no parameters input";
      pcl::Console().WriteLn( pcl::String( pmsg ) );
   }

   // ---- channel mapping ----
   // ---- CFA (raw Bayer) detection ----
   const int cfaType = view.Window().FileInfo().cfaType;
   C.isCFA = ( cfaType >= pcl::CFAType::BGGR && cfaType <= pcl::CFAType::RGGB );
   if ( C.isCFA )
   {
      if ( inS.c != 3 )
         throw pcl::Error( "fk-astro: CFA images require a 3-channel model "
                           "(or debayer the image first)." );
      switch ( cfaType )
      {
      case pcl::CFAType::GRBG: C.cfaMap[0][0]=1; C.cfaMap[0][1]=0; C.cfaMap[1][0]=2; C.cfaMap[1][1]=1; break;
      case pcl::CFAType::GBRG: C.cfaMap[0][0]=1; C.cfaMap[0][1]=2; C.cfaMap[1][0]=0; C.cfaMap[1][1]=1; break;
      case pcl::CFAType::BGGR: C.cfaMap[0][0]=2; C.cfaMap[0][1]=1; C.cfaMap[1][0]=1; C.cfaMap[1][1]=0; break;
      default: /* RGGB */      C.cfaMap[0][0]=0; C.cfaMap[0][1]=1; C.cfaMap[1][0]=1; C.cfaMap[1][1]=2; break;
      }
      C.N = 3;              // logical RGB processing; write-back re-mosaics
      C.monoModel = false;
      C.replicate = false;
      if ( p_verbose )
         pcl::Console().WriteLn( pcl::IsoString().Format(
            "fk-astro: CFA input detected (pattern %d): per-tile demosaic -> "
            "RGB -> re-mosaic", cfaType ).c_str() );
   }
   else
   {
      if ( cfaType == pcl::CFAType::CYGM )
         throw pcl::Error( "fk-astro: CYGM CFA images are not supported." );
      C.monoModel = ( inS.c == 1 && C.N > 1 );     // mono model: per-channel passes
      C.replicate = ( inS.c > 1 && C.N == 1 );     // color model on mono image
   }
   if ( !C.monoModel && !C.replicate && !C.isCFA && inS.c != C.N )
      throw pcl::Error( pcl::IsoString().Format(
         "fk-astro: the model expects %d input channel(s); the image has %d channel(s).",
         inS.c, C.N ) );

   // ---- 几何: FKv3 分块网格 ----
   //   pitch = D/n, 原点按 llround 公式; 不再是固定步长 + 末尾补齐。
   if ( outS.h > inS.h || outS.w > inS.w )
      throw pcl::Error( "fk-astro: this model enlarges the image, which FKv3 "
                        "does not support." );

   C.ctx = ( inS.h - outS.h ) / 2;
   if ( C.ctx < 0 )
      C.ctx = 0;

   const double frac = std::min( std::max( p_tileOverlap, 0.0 ), 0.5 );
   C.gx = fka::fkv3::AxisGrid( C.W, outS.w, frac );
   C.gy = fka::fkv3::AxisGrid( C.H, outS.h, frac );

   if ( p_verbose )
   {
      pcl::IsoString summary;
      summary.Format( "fk-astro: window %dx%dx%d -> center %dx%d | ctx %d | "
                      "tiles %dx%d pitch %.2f,%.2f | overlap %d | image %dx%dx%d",
                      inS.c, inS.h, inS.w, outS.w, outS.h, C.ctx,
                      int( C.gx.Count() ), int( C.gy.Count() ),
                      double( C.gx.pitch ), double( C.gy.pitch ),
                      C.gx.overlap, C.N, C.H, C.W );
      pcl::Console().WriteLn( pcl::String( summary ) );
   }

   try
   {
      if ( image.IsFloatSample() )
         switch ( image.BitsPerSample() )
         {
         case 32: RunPipeline( static_cast<pcl::Image&>( *image ), C ); break;
         case 64: RunPipeline( static_cast<pcl::DImage&>( *image ), C ); break;
         default: throw pcl::Error( "fk-astro: unsupported floating point sample format." );
         }
      else
         switch ( image.BitsPerSample() )
         {
         case 8:  RunPipeline( static_cast<pcl::UInt8Image&>( *image ), C ); break;
         case 16: RunPipeline( static_cast<pcl::UInt16Image&>( *image ), C ); break;
         case 32: RunPipeline( static_cast<pcl::UInt32Image&>( *image ), C ); break;
         default: throw pcl::Error( "fk-astro: unsupported integer sample format." );
         }
   }
   catch ( pcl::Error& )
   {
      throw;
   }
   catch ( const std::exception& e )
   {
      throw pcl::Error( ( std::string( "fk-astro: std exception in pipeline: " )
                          + e.what() ).c_str() );
   }
   catch ( ... )
   {
      throw pcl::Error( "fk-astro: unknown exception in pipeline "
                        "(outside CoreML prediction)" );
   }

   if ( p_verbose )
   {
      double sec = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t0 ).count();
      pcl::Console().WriteLn( pcl::IsoString().Format(
         "fk-astro: done in %.1f s", sec ).c_str() );
   }

   if ( p_verbose && engine.EffectiveDevice() != int( p_device ) )
   {
      const char* devName = "?";
      switch ( engine.EffectiveDevice() )
      {
      case fka::DeviceAuto: devName = "Auto"; break;
      case fka::DeviceGPU:  devName = "GPU"; break;
      case fka::DeviceNE:   devName = "CPU+ANE"; break;
      case fka::DeviceCPU:  devName = "CPU"; break;
      }
      pcl::Console().WriteLn( pcl::IsoString().Format(
         "fk-astro: note: compute device fell back to %s", devName ).c_str() );
   }

   return true;
}

void* FKAstroInferenceInstance::LockParameter( const pcl::MetaParameter* p, pcl::size_type /*tableRow*/ )
{
   if ( p == TheFKModelPathParameter )    return &p_modelPath;   if ( p == TheFKSharpenStarsParameter ) return &p_sharpenStars;
   if ( p == TheFKAdjustStarHalosParameter ) return &p_adjustStarHalos;
   if ( p == TheFKSharpenNonstellarParameter ) return &p_sharpenNonstellar;
   if ( p == TheFKAutoPSFParameter ) return &p_autoPSF;
   if ( p == TheFKPSFDiameterParameter ) return &p_psfDiameter;
   if ( p == TheFKCorrectOnlyParameter ) return &p_correctOnly;
   if ( p == TheFKNormSchemeParameter ) return &p_normScheme;
   if ( p == TheFKReflectPadParameter ) return &p_reflectPad;
   if ( p == TheFKTileOverlapParameter )  return &p_tileOverlap;
   if ( p == TheFKDeviceParameter )       return &p_device;
   if ( p == TheFKVerboseParameter )      return &p_verbose;
   return pcl::ProcessImplementation::LockParameter( p, 0 );
}
