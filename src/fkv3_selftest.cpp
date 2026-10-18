// ---------------------------------------------------------------------------
// fkv3_selftest.cpp —— FKv3 管线的离线自检(不需要 PixInsight, 不需要 PCL)
//
// 目的: 在脱离 PixInsight 的情况下, 把 C++ 的 FKv3 实现与已验证的
// Python 参考实现(python/fkv3.py)逐像素对比。测试路径与生产路径共用
// 同一份 FKv3Pipeline.h, 因此这里测过的数学就是模块里跑的那份。
//
// 用法:
//   fkv3_selftest.exe <model.onnx> <out前缀> <W> <H> <C> [overlap] [mode]
//        overlap : 0 .. 0.5, 默认 0.2
//        mode    : co | sh, 默认 co(correct-only)
//
// 产物:
//   <前缀>_in.f32    输入图像, 平面布局 (C,H,W), float32
//   <前缀>_out.f32   输出图像, 同上
//   stdout           归一化参数与分块数, 供对比
// ---------------------------------------------------------------------------

#include "FKv3Pipeline.h"
#include "InferenceEngine.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace
{

// 确定性伪随机(不依赖标准库分布, 保证跨平台一致)
struct Lcg
{
   uint64_t s;
   explicit Lcg( uint64_t seed ) : s( seed ) {}
   double Next()
   {
      s = s * 6364136223846793005ULL + 1442695040888963407ULL;
      return double( ( s >> 11 ) & 0x1FFFFFFFFFFFFFULL ) / double( 0x1FFFFFFFFFFFFFULL );
   }
   double Gaussian()
   {
      // Box-Muller, 只用一次 cos
      const double u1 = std::max( 1.0e-12, Next() );
      const double u2 = Next();
      return std::sqrt( -2.0 * std::log( u1 ) ) * std::cos( 6.283185307179586 * u2 );
   }
};

// 合成一张有背景梯度 + 星点 + 噪声的图, 平面布局 (C,H,W)
std::vector<float> MakeSynthetic( int W, int H, int C, double background )
{
   std::vector<float> img( size_t( W ) * size_t( H ) * size_t( C ), 0.0f );
   Lcg rng( 0x5EED1234ULL );

   // 星点位置与亮度(固定, 三个通道共用 -> 灰度星点, 便于观察通道一致性)
   struct Star { double x, y, amp, sigma; };
   std::vector<Star> stars;
   const int nStars = std::max( 6, ( W * H ) / 20000 );
   for ( int i = 0; i < nStars; ++i )
      stars.push_back( { rng.Next() * W, rng.Next() * H,
                         0.02 + 0.35 * rng.Next(), 1.2 + 2.2 * rng.Next() } );

   for ( int c = 0; c < C; ++c )
   {
      float* plane = img.data() + size_t( c ) * size_t( W ) * size_t( H );
      const double cScale = 1.0 + 0.12 * c;   // 通道间轻微亮度差
      for ( int y = 0; y < H; ++y )
      {
         const double grad = 0.35 * double( y ) / double( std::max( 1, H - 1 ) );
         for ( int x = 0; x < W; ++x )
         {
            double v = background * ( 1.0 + grad ) * cScale;
            v += 0.004 * rng.Gaussian();                     // 背景噪声
            for ( const Star& s : stars )
            {
               const double dx = x - s.x, dy = y - s.y;
               const double r2 = dx * dx + dy * dy;
               if ( r2 < 36.0 * s.sigma * s.sigma )
                  v += s.amp * cScale * std::exp( -0.5 * r2 / ( s.sigma * s.sigma ) );
            }
            plane[size_t( y ) * size_t( W ) + size_t( x )] = float( v );
         }
      }
   }
   return img;
}

bool WriteF32( const std::string& path, const std::vector<float>& data )
{
   FILE* f = std::fopen( path.c_str(), "wb" );
   if ( !f )
      return false;
   const size_t n = std::fwrite( data.data(), sizeof( float ), data.size(), f );
   std::fclose( f );
   return n == data.size();
}

} // namespace

int main( int argc, char** argv )
{
   if ( argc < 6 )
   {
      std::printf( "usage: %s <model.onnx> <outprefix> <W> <H> <C> [overlap] [co|sh]\n", argv[0] );
      return 2;
   }

   const std::string model = argv[1];
   const std::string prefix = argv[2];
   const int W = std::atoi( argv[3] );
   const int H = std::atoi( argv[4] );
   const int C = std::atoi( argv[5] );
   const double overlap = ( argc > 6 ) ? std::atof( argv[6] ) : 0.2;
   const std::string mode = ( argc > 7 ) ? argv[7] : "co";
   const bool correctOnly = ( mode != "sh" );

   if ( W < 1 || H < 1 || C < 1 )
   {
      std::printf( "bad dimensions\n" );
      return 2;
   }

   // ---- 1. 引擎 ----
   fka::InferenceEngine& eng = fka::InferenceEngine::Get();
   if ( !eng.Load( model, fka::DeviceAuto ) )
   {
      std::printf( "ENGINE ERROR: %s\n", eng.LastError().c_str() );
      return 1;
   }
   const fka::TensorShape inS = eng.InputShape();
   const fka::TensorShape outS = eng.OutputShape();
   std::printf( "engine   ORT %s  in=%dx%dx%d out=%dx%dx%d params=%zu\n",
                eng.RuntimeVersion().c_str(),
                inS.c, inS.h, inS.w, outS.c, outS.h, outS.w, eng.ParamsSize() );
   if ( !inS.IsValid() || !outS.IsValid() || inS.h != inS.w || outS.h != outS.w )
   {
      std::printf( "unexpected model shape\n" );
      return 1;
   }
   const int tileIn = inS.h, tileOut = outS.h;

   // ---- 2. 合成输入并落盘 ----
   const double background = 0.006;   // 典型深空背景
   std::vector<float> img = MakeSynthetic( W, H, C, background );
   if ( !WriteF32( prefix + "_in.f32", img ) )
   {
      std::printf( "cannot write %s_in.f32\n", prefix.c_str() );
      return 1;
   }

   // ---- 3. 归一化: 整图 Fit 一次 ----
   fka::fkv3::ArcsinhNormalizer nz;
   nz.Fit( W, H, C, /*lunar*/ false,
           [&]( int c, int y, int x ) -> double
           {
              return double( img[ size_t( c ) * size_t( W ) * size_t( H )
                                  + size_t( y ) * size_t( W ) + size_t( x ) ] );
           } );
   std::printf( "normalizer low=%.12g high=%.12g median=%.12g x=%.12g gain=%.12g\n",
                nz.low, nz.high, nz.median, nz.x, nz.gain );
   std::printf( "           crop=%d,%d %dx%d grid=%dx%d positives=%zu\n",
                nz.cropX, nz.cropY, nz.cropW, nz.cropH, nz.gridW, nz.gridH,
                nz.positiveSamples );

   // ---- 4. 整图 Forward(一次), 单色复制到 3 个模型通道 ----
   const int Cfeed = ( C == 1 ) ? 3 : C;
   std::vector<double> feed( size_t( W ) * size_t( H ) * size_t( Cfeed ), 0.0 );
   for ( int c = 0; c < Cfeed; ++c )
   {
      const int srcC = ( C == 1 ) ? 0 : c;
      const float* sp = img.data() + size_t( srcC ) * size_t( W ) * size_t( H );
      double* dp = feed.data() + size_t( c ) * size_t( W ) * size_t( H );
      for ( size_t i = 0, n = size_t( W ) * size_t( H ); i < n; ++i )
         dp[i] = nz.Forward( double( sp[i] ) );
   }

   // ---- 5. 分块 ----
   fka::fkv3::AxisGrid gx( W, tileOut, overlap );
   fka::fkv3::AxisGrid gy( H, tileOut, overlap );
   std::printf( "tiling   overlap=%.3f tiles=%zux%zu pitch=%.6f,%.6f\n",
                overlap, gx.Count(), gy.Count(), gx.pitch, gy.pitch );

   std::vector<std::vector<double>> wx( gx.Count() ), wy( gy.Count() );
   for ( size_t i = 0; i < gx.Count(); ++i ) wx[i] = gx.Window( i );
   for ( size_t j = 0; j < gy.Count(); ++j ) wy[j] = gy.Window( j );

   const int ctx = ( tileIn - tileOut ) / 2;
   const size_t plane = size_t( W ) * size_t( H );

   std::vector<std::vector<double>> acc( size_t( outS.c ), std::vector<double>( plane, 0.0 ) );
   std::vector<double> wsum( plane, 0.0 );

   std::vector<float> inBuf( size_t( Cfeed ) * size_t( tileIn ) * size_t( tileIn ) );
   std::vector<float> outBuf( size_t( outS.c ) * size_t( tileOut ) * size_t( tileOut ) );

   // parameters 张量
   fka::fkv3::Settings st;
   st.correctOnly = correctOnly;
   st.stars = correctOnly ? 0.0 : 0.5;
   st.nonstellar = correctOnly ? 0.0 : 0.5;
   const std::array<float, 6> pv = st.Parameters( nz.gain );
   std::printf( "params   [%.6f %.6f %.6f %.6f %.6f %.6f]\n",
                pv[0], pv[1], pv[2], pv[3], pv[4], pv[5] );

   for ( size_t j = 0; j < gy.Count(); ++j )
   {
      const int y0 = gy.origins[j];
      const int cy0 = y0 - ctx;
      for ( size_t i = 0; i < gx.Count(); ++i )
      {
         const int x0 = gx.origins[i];
         const int cx0 = x0 - ctx;

         // 输入块: 越界区域保持 0(模型域), 不是 Forward(0)
         std::fill( inBuf.begin(), inBuf.end(), 0.0f );
         for ( int c = 0; c < Cfeed; ++c )
         {
            const double* sp = feed.data() + size_t( c ) * plane;
            float* dp = inBuf.data() + size_t( c ) * size_t( tileIn ) * size_t( tileIn );
            const int iy0 = std::max( 0, cy0 ), iy1 = std::min( H, cy0 + tileIn );
            const int ix0 = std::max( 0, cx0 ), ix1 = std::min( W, cx0 + tileIn );
            for ( int y = iy0; y < iy1; ++y )
            {
               const double* row = sp + size_t( y ) * size_t( W );
               float* drow = dp + size_t( y - cy0 ) * size_t( tileIn ) + size_t( ix0 - cx0 );
               for ( int x = ix0; x < ix1; ++x )
                  drow[size_t( x - ix0 )] = float( row[size_t( x )] );
            }
         }

         if ( !eng.Predict( inBuf.data(), outBuf.data(), pv.data(), pv.size() ) )
         {
            std::printf( "PREDICT ERROR: %s\n", eng.LastError().c_str() );
            return 1;
         }

         // 融合: double 累加, 不除权重和
         fka::fkv3::BlendTile( outBuf.data(), outS.c, ( C == 1 ) ? 1 : C,
                               tileOut, tileOut, x0, y0, W, H, wx[i], wy[j],
                               [&]( int c, size_t dst, double v )
                               {
                                  acc[size_t( c )][dst] += v;
                               } );

         // 权重和(仅诊断用, 不参与除法)
         const int yBegin = std::max( 0, y0 ), yEnd = std::min( H, y0 + tileOut );
         const int xBegin = std::max( 0, x0 ), xEnd = std::min( W, x0 + tileOut );
         for ( int y = yBegin; y < yEnd; ++y )
            for ( int x = xBegin; x < xEnd; ++x )
               wsum[size_t( y ) * size_t( W ) + size_t( x )]
                  += wx[i][size_t( x - x0 )] * wy[j][size_t( y - y0 )];
      }
   }

   // ---- 6. 整图一次逆变换 + clamp ----
   std::vector<float> result( size_t( C ) * plane );
   for ( size_t idx = 0; idx < plane; ++idx )
   {
      if ( C == 1 )
      {
         // BlendTile 在单色模式下已经把三通道按 (R+G)/3 + B/3 合并进 acc[0],
         // 这里不能再除一次 3。
         double v = nz.Inverse( acc[0][idx] );
         if ( v < 0.0 ) v = 0.0; else if ( v > 1.0 ) v = 1.0;
         result[idx] = float( v );
      }
      else
         for ( int c = 0; c < C; ++c )
         {
            double v = nz.Inverse( acc[size_t( c )][idx] );
            if ( v < 0.0 ) v = 0.0; else if ( v > 1.0 ) v = 1.0;
            result[ size_t( c ) * plane + idx ] = float( v );
         }
   }

   double wmin = wsum.empty() ? 0.0 : wsum[0], wmax = wmin;
   for ( double v : wsum ) { wmin = std::min( wmin, v ); wmax = std::max( wmax, v ); }
   std::printf( "wsum     min=%.9g max=%.9g\n", wmin, wmax );

   if ( !WriteF32( prefix + "_out.f32", result ) )
   {
      std::printf( "cannot write %s_out.f32\n", prefix.c_str() );
      return 1;
   }
   std::printf( "done     %s_out.f32  (%zu samples)\n", prefix.c_str(), result.size() );
   return 0;
}
