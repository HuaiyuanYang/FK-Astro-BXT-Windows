/*
 * fk-astro — PixInsight module entry points
 *
 * A skeleton PixInsight module that runs user-trained CoreML models
 * (.mlpackage / .mlmodel) on images, with tiled inference.
 */

#include <pcl/MetaModule.h>
#include <pcl/MetaProcess.h>

#include "FKAstroProcess.h"
#include "FKAstroInterface.h"

// ---------------------------------------------------------------------------

class FKAstroModule : public pcl::MetaModule
{
public:

   const char* Version() const override
   {
      return PCL_MODULE_VERSION_S( 01, 00, 00, 0001, eng, release );
   }

   pcl::IsoString Name() const override
   {
      return "FKAstro";
   }

   pcl::String Description() const override
   {
      return "BXT: run the converted bxt ONNX model in PixInsight (Windows / ONNX Runtime).";
   }

   pcl::String Company() const override
   {
      return "fk-astro project";
   }

   pcl::String Author() const override
   {
      return "FKAstro";
   }

   pcl::String Copyright() const override
   {
      return "Copyright (c) 2026 fk-astro. All rights reserved.";
   }

   pcl::String TradeMarks() const override
   {
      return pcl::String();
   }

   void GetReleaseDate( int& year, int& month, int& day ) const override
   {
      year = 2026; month = 10; day = 12;
   }
};

// ---------------------------------------------------------------------------

extern "C" PCL_MODULE_EXPORT int32_t InstallPixInsightModule( int32_t /*mode*/ )
{
   /*
    * N.B. The MetaModule constructor installs the new module object as the
    * pcl::Module global singleton, and each MetaProcess constructor attaches
    * to it. This must happen for every install mode (full install, module
    * query and verification), hence the one-shot guard below.
    */
   static bool installed = false;
   if ( !installed )
   {
      installed = true;
      new FKAstroModule();
      new FKAstroInferenceProcess();
      TheFKAstroInterface = new FKAstroInterface();
   }
   return 0;
}
