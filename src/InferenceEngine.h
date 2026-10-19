#pragma once

/*
 * InferenceEngine —— PixInsight 模块的 ONNX Runtime 推理引擎(Windows)
 *
 * ── 这份文件是什么 ────────────────────────────────────────────────────
 * fk-astro 原版的 CoreMLEngine.h/.mm 是 macOS + CoreML 实现。
 * 本文件提供**公开接口逐字相同**的 Windows + ONNX Runtime 实现, 因此
 * fk-astro 的 FKAstroProcess.cpp / FKAstroInterface.cpp / FKAstroModule.cpp
 * 只需把 `#include "CoreMLEngine.h"` 改成 `#include "InferenceEngine.h"`,
 * 其余一行都不用动(类名兼容别名见文件末尾)。
 *
 * ── 为什么动态加载 onnxruntime.dll ─────────────────────────────────────
 * PixInsight 自带 bin\onnxruntime.dll(本机实测 1.25.20260427, ORT_API_VERSION 25),
 * 所以我们:
 *     * 不 import .lib        —— 不需要 ORT 的导入库
 *     * 不打包 ORT           —— 不增加部署体积, 也不会和 PI 版本打架
 *     * 运行时协商版本       —— OrtApiBase::GetApi(ORT_API_VERSION), PI 升级 ORT 也不会崩
 * DLL 位置: 本模块所在目录下的 onnxruntime.dll(即 PI 的 bin\), 可用环境变量
 * BXT_ORT_DLL 覆盖(便于脱离 PI 做单元测试)。
 *
 * ── 与 CoreML 版的语义差异(已在实现里处理)────────────────────────────
 *   1. 本项目的模型输入是 4-D `[1,H,W,C]`(NHWC), 而 Predict 契约是 CHW;
 *      引擎内部转置, 调用方看不到差别。
 *   2. ComputeDevice 里的 DeviceNE 是 Apple ANE, Windows 无对应物;
 *      请求它等同于 DeviceAuto(默认 EP)。
 *   3. 参数张量是 `[1,N]`, 直接喂。
 */

#include <string>
#include <vector>
#include <cstdint>

namespace fka
{

struct TensorShape
{
   int c = 0;
   int h = 0;
   int w = 0;

   bool IsValid() const
   {
      return c > 0 && h > 0 && w > 0;
   }

   size_t Size() const
   {
      return size_t( c ) * size_t( h ) * size_t( w );
   }
};

enum ComputeDevice
{
   DeviceAuto = 0,   // 由 ORT 决定(默认 EP 集合)
   DeviceGPU  = 1,   // 请求 GPU EP(CUDA / DirectML), 不可用时回落
   DeviceNE   = 2,   // Apple ANE, Windows 无对应物 -> 等同 DeviceAuto
   DeviceCPU  = 3    // 仅 CPU
};

class InferenceEngine
{
public:

   static InferenceEngine& Get();

   /*
    * 加载模型(path 为 .onnx 的文件系统路径)。
    * 返回 true 表示成功; 失败时 LastError() 带诊断信息。
    */
   bool Load( const std::string& path, int device );

   bool IsLoaded() const
   {
      return m_loaded;
   }

   /*
    * 模型第一个输入的形状, **已归一化成 {c,h,w}**。
    * 4-D 输入 `[1,H,W,C]` 会被识别为 NHWC 并还原成 c=C,h=H,w=W。
    */
   TensorShape InputShape() const
   {
      return m_inputShape;
   }

   /*
    * 模型第一个输出的形状, 同样归一化成 {c,h,w}。
    */
   TensorShape OutputShape() const
   {
      return m_outputShape;
   }

   /*
    * 模型是否声明了第二个输入(参数张量, 形如 [1,5] / [1,6])。
    */
   bool HasParamsInput() const
   {
      return m_paramsSize > 0;
   }

   size_t ParamsSize() const
   {
      return m_paramsSize;
   }

   /*
    * 跑一次推理。
    *
    * 'in'     —— 输入样本, **CHW 布局**: in[c*h*w + y*w + x]
    * 'out'    —— 输出样本, 同样 CHW 布局, 至少 OutputShape().Size() 个 float
    * 'params' —— 第二个输入(参数张量)的值; 多出的元素补 0, 模型没有该输入时忽略
    *
    * 返回 true 表示成功。
    */
   bool Predict( const float* in, float* out,
                 const float* params = nullptr, size_t paramCount = 0 );

   const std::string& LastError() const
   {
      return m_lastError;
   }

   /*
    * 实际生效的设备(回落之后的值)。
    */
   int EffectiveDevice() const
   {
      return m_effectiveDevice;
   }

   /*
    * 实际被追加进会话的执行器名, 例如 "CUDAExecutionProvider"。
    * 空字符串 = 只用 ORT 的默认 CPU 执行器。
    */
   const std::string& ActiveProvider() const
   {
      return m_activeProvider;
   }

   void Unload();

   /*
    * 诊断用: ORT 版本字符串 + 该 ORT 构建可用的 provider 列表。
    * 载入 DLL 失败时为空。
    */
   const std::string& RuntimeVersion() const
   {
      return m_runtimeVersion;
   }

   const std::vector<std::string>& AvailableProviders() const
   {
      return m_providers;
   }

private:

   InferenceEngine() = default;
   ~InferenceEngine();

   InferenceEngine( const InferenceEngine& ) = delete;
   InferenceEngine& operator=( const InferenceEngine& ) = delete;

   bool        m_loaded = false;
   std::string m_loadedPath;
   int         m_device = DeviceAuto;
   int         m_effectiveDevice = DeviceAuto;
   TensorShape m_inputShape;
   TensorShape m_outputShape;
   std::string m_lastError;
   std::string m_runtimeVersion;
   std::string m_activeProvider;      // 实际追加的执行器, 空 = CPU
   std::vector<std::string> m_providers;

   // 运行时打开的 ORT(不透明指针, 避免把头文件泄漏给调用方)
   void* m_ortModule   = nullptr;   // HMODULE
   void* m_ortApi      = nullptr;   // const OrtApi*
   void* m_env         = nullptr;   // OrtEnv*
   void* m_session     = nullptr;   // OrtSession*
   void* m_memoryInfo  = nullptr;   // OrtMemoryInfo*

   std::string m_inputName;
   std::string m_outputName;
   std::string m_paramsName;        // 第二个输入, 可能为空

   bool   m_inputNHWC  = true;      // 模型输入是 NHWC(需转置)
   bool   m_outputNHWC = true;
   int    m_inputRank  = 4;
   size_t m_paramsSize = 0;

   // 模型声明的原始维度(建张量时直接透传, 只对数据做转置)
   std::vector<int64_t> m_inputDims;
   std::vector<int64_t> m_outputDims;
   std::vector<int64_t> m_paramsDims;

   // 复用的暂存缓冲(按块调用, 避免每块都分配)
   std::vector<float> m_inStage;      // NHWC 输入暂存
   std::vector<float> m_paramsStage;  // 参数张量暂存

   std::string TakeError( void* status );
   bool        EnsureRuntime();
   void        ReleaseAll();
};

// 兼容别名: 让 fk-astro 的现有源码在只改 include 的情况下继续编译
using CoreMLEngine = InferenceEngine;

} // namespace fka
