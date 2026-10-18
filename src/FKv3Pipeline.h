#pragma once

// ---------------------------------------------------------------------------
// FKv3 宿主管线 —— 归一化 / 分块 / 融合
//
// 这份头文件不依赖 PCL, 也不依赖 onnxruntime, 因此可以脱离 PixInsight
// 单独编译和单元测试(fkv3_selftest.cpp)。生产路径与测试路径共用它,
// 保证"测过的就是跑的那份"。
//
// 数学行为依据 docs/FKv3.md(从官方二进制逆向得到的规格);
// 与 python/fkv3.py 逐项对齐, 后者已用真实数据与官方 BXT 对比验证过
// (P99 < 2e-4, 相关 > 0.999)。
//
// 与旧方案 FKv1 的四处关键差异:
//   1. 归一化是**整图** arcsinh, 不是逐块 log+MTF+仿射
//   2. 参数第 6 列是 gain = log10(x/a), 不是 -log10(bg)
//   3. 分块用 pitch = D/n 与 llround 原点, 不是固定步长
//   4. 融合用余弦带且**不除以权重和**, 整图只做一次逆变换
// ---------------------------------------------------------------------------

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace fka
{
namespace fkv3
{

// ---------------------------------------------------------------------------
// 样本 <-> double
//
// 整数样本除以类型最大值(16 位图即 65535), 与 PixInsight 的浮点图像约定一致。
// 写回时用 nearbyint(就近取偶), 与 PCL 的转换一致, 不是截断。
// ---------------------------------------------------------------------------

template <class S>
inline double SampleToDouble( S value )
{
   if constexpr ( std::is_floating_point<S>::value )
      return double( value );
   else
      return double( value ) / double( std::numeric_limits<S>::max() );
}

template <class S>
inline S SampleFromDouble( double value )
{
   if constexpr ( std::is_floating_point<S>::value )
      return S( value );
   else
      return S( std::nearbyint( value * double( std::numeric_limits<S>::max() ) ) );
}

// ---------------------------------------------------------------------------
// 用户设置
// ---------------------------------------------------------------------------

struct Settings
{
   double stars = 0.5;        // 星点锐化强度        0 .. 0.7
   double halos = 0.0;        // 星点光晕           -0.5 .. 0.5
   double diameter = 0.0;     // 手动 PSF 直径(像素)  0 .. 8
   double nonstellar = 0.5;   // 非星点锐化强度      0 .. 1.0
   bool   autoPSF = true;     // 自动估计 PSF
   bool   correctOnly = false;// 只校正, 不锐化
   bool   lunar = false;      // 月球/行星模式

   // 生效值: 按官方顺序逐层屏蔽, 顺序不能换。
   Settings Effective() const
   {
      Settings r = *this;
      if ( correctOnly )
      {
         r.stars = r.halos = r.diameter = r.nonstellar = 0.0;
         r.autoPSF = true;
         r.lunar = false;
      }
      if ( r.lunar )
      {
         r.stars = r.halos = 0.0;
         r.autoPSF = false;
      }
      if ( r.autoPSF )
         r.diameter = 0.0;
      return r;
   }

   // 模型的 parameters 张量, 形状 [1,6]。
   std::array<float, 6> Parameters( double gain ) const
   {
      const Settings e = Effective();
      return { float( e.stars ),
               float( e.halos ),
               float( e.diameter * 0.5 ),   // 直径折半后喂给模型
               float( e.autoPSF ? 1 : 0 ),
               float( e.nonstellar ),
               float( gain ) };
   }
};

// ---------------------------------------------------------------------------
// 整图 arcsinh 归一化
//
// 拟合一次, 之后输入像素、参数第 6 列、逆变换都复用同一份状态 ——
// 三者必须来自同一次拟合, 否则会引入系统性偏移。
// ---------------------------------------------------------------------------

struct ArcsinhNormalizer
{
   double low = 0.0;      // 低分位点(正样本的 0.1% 分位)
   double high = 1.0;     // 高分位点, arcsinh 方案固定为 1
   double median = 0.0;   // 映射后样本的上中位数
   double x = 1.0;        // arcsinh 强度系数
   double asinhX = std::asinh( 1.0 );
   double gain = std::log10( 1.0 / std::asinh( 1.0 ) );

   int    cropX = 0, cropY = 0, cropW = 0, cropH = 0, gridW = 0, gridH = 0;
   size_t positiveSamples = 0;

   // at(c,y,x) 读取一个原始样本(已是 double, 未归一化)。
   template <class PixelAt>
   void Fit( int width, int height, int channels, bool lunar, PixelAt at )
   {
      *this = ArcsinhNormalizer();
      if ( width < 1 || height < 1 || channels < 1 )
         throw std::invalid_argument( "FKv3: 归一化需要非空图像" );

      // 1) 四边各裁 5%, 裁废则回退整图
      cropX = int( width * 0.05 );
      cropY = int( height * 0.05 );
      cropW = width - 2 * cropX;
      cropH = height - 2 * cropY;
      if ( cropW < 1 || cropH < 1 )
      {
         cropX = cropY = 0;
         cropW = width;
         cropH = height;
      }

      // 2) 重采样到长边 128(尺寸四舍五入, 取点向下取整)
      const double scale = double( std::max( cropW, cropH ) ) / 128.0;
      gridW = std::max( 1, int( std::llround( cropW / scale ) ) );
      gridH = std::max( 1, int( std::llround( cropH / scale ) ) );
      const double stepX = double( cropW ) / gridW;
      const double stepY = double( cropH ) / gridH;

      // 3) 各通道正样本合并
      std::vector<double> samples;
      samples.reserve( size_t( gridW ) * size_t( gridH ) * size_t( channels ) );
      for ( int c = 0; c < channels; ++c )
         for ( int gy = 0; gy < gridH; ++gy )
         {
            const int sy = cropY + std::min( cropH - 1, int( std::floor( gy * stepY ) ) );
            for ( int gx = 0; gx < gridW; ++gx )
            {
               const int sx = cropX + std::min( cropW - 1, int( std::floor( gx * stepX ) ) );
               const double v = at( c, sy, sx );
               if ( v > 0.0 )
                  samples.push_back( v );
            }
         }
      positiveSamples = samples.size();

      // 4) 低分位点: 位置 (n-1)*0.001 的线性插值
      //    高分位点保持 1(arcsinh 方案的 wp = 1)
      if ( !samples.empty() )
      {
         std::sort( samples.begin(), samples.end() );
         const double q = double( samples.size() - 1 ) * 0.001;
         const size_t i = size_t( q );
         const size_t j = std::min( i + 1, samples.size() - 1 );
         low = samples[i] + ( q - double( i ) ) * ( samples[j] - samples[i] );
      }

      // 5) 分位点跨度太小说明是极端图像, 退回单位映射
      if ( std::fabs( high - low ) < 0.001 )
      {
         low = 0.0;
         high = 1.0;
      }

      // 6) 映射后取上中位数, 反解 arcsinh 强度
      const double invSpan = 1.0 / ( high - low );
      for ( double& v : samples )
         v = ( v - low ) * invSpan;

      if ( !samples.empty() )
      {
         const size_t mid = samples.size() / 2;   // 偶数个时取偏上的那个
         std::nth_element( samples.begin(), samples.begin() + ptrdiff_t( mid ), samples.end() );
         median = samples[mid];
         if ( !lunar && median > 1.0e-6 )
         {
            const double denom = median * ( median + 0.1 ) + 1.0e-12;
            x = std::min( 1.0e6, std::max( 1.0, 0.2 / denom ) );
         }
      }

      asinhX = std::asinh( x );
      gain = std::log10( x / asinhX );
   }

   // 前向: 不裁剪、不丢弃负值 —— 负样本同样参与映射。
   double Forward( double value ) const
   {
      return std::asinh( ( value - low ) * ( 1.0 / ( high - low ) ) * x ) / asinhX;
   }

   double Inverse( double value ) const
   {
      return low + ( std::sinh( value * asinhX ) / x ) * ( high - low );
   }
};

// ---------------------------------------------------------------------------
// 单轴分块网格
//
// 接缝位置由 pitch = D/n 决定, 而不是由固定步长决定 —— 这是 FKv3 与旧方案
// 在接缝可见性上的根本差别。
// ---------------------------------------------------------------------------

struct AxisGrid
{
   int   tile = 0;          // 输出块尺寸
   int   overlap = 0;       // 重叠像素数
   float pitch = 0.0f;      // 块间距(浮点, 参与接缝计算)
   std::vector<int> origins;

   AxisGrid() = default;
   AxisGrid( int size, int tileSize, double fraction ) : tile( tileSize )
   {
      if ( size < 1 || tile < 1 )
         throw std::invalid_argument( "FKv3: 分块尺寸必须为正" );

      // 重叠数向下取到偶数, 保证接缝两侧对称
      const float f = float( std::min( 0.5, std::max( 0.0, fraction ) ) );
      overlap = 2 * int( std::floor( float( tile ) * f * 0.5f ) );

      const int count = size <= tile
                        ? 1
                        : int( std::ceil( double( size ) / double( tile - overlap ) ) );
      pitch = float( double( size ) / double( count ) );

      origins.reserve( size_t( count ) );
      for ( int k = 0; k < count; ++k )
      {
         int origin;
         if ( size < tile )
            origin = ( size - tile ) / 2;   // 有符号除法, 原点可能为负
         else if ( k == 0 )
            origin = 0;
         else if ( k == count - 1 )
            origin = size - tile;
         else
         {
            // 必须先在 float 下做 pitch*k 与 T-pitch, 再提升到 double 相减
            const float a = pitch * float( k );
            const float b = float( tile ) - pitch;
            origin = int( std::llround( double( a ) - 0.5 * double( b ) ) );
         }
         origins.push_back( origin );
      }
   }

   size_t Count() const { return origins.size(); }

   // 第 index 块的余弦带权重(长度为 tile)。
   // 只有与相邻块相接的一侧才衰减; 整图外边界不衰减。
   std::vector<double> Window( size_t index ) const
   {
      constexpr double pi = 3.14159265358979323846;
      std::vector<double> w( size_t( tile ), 1.0 );
      if ( tile < 1 || index >= origins.size() )
         return w;

      const int origin = origins[index];
      auto falling = [&]( int j )
      {
         return ( std::cos( ( j + 1 ) * pi / ( overlap + 1 ) ) + 1.0 ) * 0.5;
      };

      if ( index > 0 )
      {
         const int seam = int( std::llround( double( float( pitch * float( index ) ) ) ) );
         const int left = seam - origin - overlap / 2;
         for ( int i = 0; i < tile; ++i )
            if ( i < left )
               w[size_t( i )] = 0.0;
            else if ( i < left + overlap )
               w[size_t( i )] *= 1.0 - falling( i - left );
      }

      if ( index + 1 < origins.size() )
      {
         const int seam = int( std::llround( double( float( pitch * float( index + 1 ) ) ) ) );
         const int right = seam - origin - overlap / 2;
         for ( int i = 0; i < tile; ++i )
            if ( i >= right + overlap )
               w[size_t( i )] = 0.0;
            else if ( i >= right )
               w[size_t( i )] *= falling( i - right );
      }
      return w;
   }
};

// ---------------------------------------------------------------------------
// 把一块模型输出累加进目标缓冲区
//
// output 是模型空间的原始输出(未做逆变换)。累加在 double 下进行, 且
// **不除以权重和** —— 权重带本身已保证相邻块之和趋近 1。
//
// add(channel, dstIndex, value) 负责实际的累加。
// ---------------------------------------------------------------------------

template <class Add>
void BlendTile( const float* output, int modelChannels, int imageChannels,
                int tileW, int tileH,
                int x0, int y0, int width, int height,
                const std::vector<double>& wx, const std::vector<double>& wy,
                Add add )
{
   // 小图的原点可能为负, 先把目标矩形裁到图像内再算偏移
   const int yBegin = std::max( 0, y0 );
   const int yEnd   = std::min( height, y0 + tileH );
   const int xBegin = std::max( 0, x0 );
   const int xEnd   = std::min( width, x0 + tileW );

   for ( int y = yBegin; y < yEnd; ++y )
      for ( int x = xBegin; x < xEnd; ++x )
      {
         const int ox = x - x0;
         const int oy = y - y0;
         const size_t src = size_t( oy ) * size_t( tileW ) + size_t( ox );
         const size_t dst = size_t( y ) * size_t( width ) + size_t( x );

         auto weighted = [&]( int c ) -> double
         {
            const size_t idx = size_t( c ) * size_t( tileW ) * size_t( tileH ) + src;
            return ( double( output[idx] ) * wx[size_t( ox )] ) * wy[size_t( oy )];
         };

         if ( modelChannels == 3 && imageChannels == 1 )
         {
            // 单色图: 先把 R+G 成对相加, 再与 B 一起各乘 1/3。
            // 这个分组顺序会影响末位, 必须保持。
            const double pair = weighted( 0 ) + weighted( 1 );
            constexpr double third = 1.0 / 3.0;
            add( 0, dst, pair * third + weighted( 2 ) * third );
         }
         else
         {
            const int nc = std::min( modelChannels, imageChannels );
            for ( int c = 0; c < nc; ++c )
               add( c, dst, weighted( c ) );
         }
      }
}

} // namespace fkv3
} // namespace fka
