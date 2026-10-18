#pragma once

/*
 * fk-astro — FKAstroInference custom process interface
 *
 * Section layout:
 *   Model                 — model path
 *   Stellar Adjustments   — Sharpen Stars, Adjust Star Halos
 *   Nonstellar Adjustments — Sharpen Nonstellar, Automatic PSF, PSF Diameter
 *   Options               — Correct Only, Normalization (None / FKv1)
 *   Additional Options    — Tile Overlap, Reflect Padding, Compute Device, Verbose
 */

#include <pcl/ProcessInterface.h>
#include <pcl/Edit.h>
#include <pcl/NumericControl.h>
#include <pcl/ComboBox.h>
#include <pcl/CheckBox.h>
#include <pcl/Sizer.h>
#include <pcl/Label.h>
#include <pcl/PushButton.h>
#include <pcl/SectionBar.h>
#include <pcl/Control.h>

class FKAstroInterface : public pcl::ProcessInterface
{
public:

   FKAstroInterface() = default;   // GUI is built lazily in Launch()

   pcl::IsoString Id() const override;
   pcl::MetaProcess* Process() const override;
   pcl::ProcessImplementation* NewProcess() const override;
   bool ImportProcess( const pcl::ProcessImplementation& ) override;
   bool Launch( const pcl::MetaProcess& process,
                const pcl::ProcessImplementation* instance,
                bool& dynamic, unsigned& flags ) override;

private:

   void e_EditCompleted( pcl::Edit& sender );
   void e_BrowseClicked( pcl::Button& sender, bool checked );
   void e_SharpenStars( pcl::NumericEdit& sender, double value );
   void e_AdjustStarHalos( pcl::NumericEdit& sender, double value );
   void e_SharpenNonstellar( pcl::NumericEdit& sender, double value );
   void e_AutoPSF( pcl::Button& sender, pcl::Button::check_state state );
   void e_PSFDiameter( pcl::NumericEdit& sender, double value );
   void e_CorrectOnly( pcl::Button& sender, pcl::Button::check_state state );
   void e_NormScheme( pcl::ComboBox& sender, int itemIndex );
   void e_ReflectPad( pcl::Button& sender, pcl::Button::check_state state );
   void e_Overlap( pcl::NumericEdit& sender, double value );
   void e_Device( pcl::ComboBox& sender, int itemIndex );
   void e_Verbose( pcl::Button& sender, pcl::Button::check_state state );
   void e_Section( pcl::SectionBar& sender, pcl::Control& section, bool start );

   // Apply the enable/disable cascade: when Correct Only is checked, all
   // adjustment controls are disabled; PSF diameter additionally requires
   // Automatic PSF to be unchecked.
   void UpdateEnabledStates();

   // Interface state (valid before GUI exists)
   pcl::String m_modelPath;
   double      m_sharpenStars      = 0.5;   // [0, 0.7]
   double      m_adjustStarHalos   = 0.0;   // [-0.5, 0.5]
   double      m_sharpenNonstellar = 0.5;   // [0, 1]
   bool        m_autoPSF      = true;
   double      m_psfDiameter  = 0.0;         // [0, 8]
   bool        m_correctOnly  = false;   // 默认关: 与 BXT 一致, 首次使用应先手动打开
   int         m_normScheme   = 1;    // 0=None 1=FKv3 —— 默认 FKv3
   bool        m_reflectPad   = true;
   double      m_tileOverlap  = 0.2;         // 重叠比例 [0, 0.5]; 0.2 是 FKv3 的标准值
   int         m_device       = 0;
   bool        m_verbose      = true;

   // GUI, allocated on first Launch()
   struct GUIData
   {
      // Model
      pcl::SectionBar      ModelBar;
      pcl::Control         ModelSection;
      pcl::VerticalSizer   ModelSizer;
      pcl::Label           ModelPathLabel;
      pcl::Edit            ModelPathEdit;
      pcl::PushButton      BrowseButton;
      pcl::HorizontalSizer PathSizer;

      // Stellar Adjustments
      pcl::SectionBar     StellarBar;
      pcl::Control        StellarSection;
      pcl::VerticalSizer  StellarSizer;
      pcl::NumericControl SharpenStars;
      pcl::NumericControl AdjustStarHalos;

      // Nonstellar Adjustments
      pcl::SectionBar     NonstellarBar;
      pcl::Control        NonstellarSection;
      pcl::VerticalSizer  NonstellarSizer;
      pcl::NumericControl SharpenNonstellar;
      pcl::CheckBox       AutoPSFCheck;
      pcl::NumericControl PSFDiameter;

      // Options
      pcl::SectionBar     OptionsBar;
      pcl::Control        OptionsSection;
      pcl::VerticalSizer  OptionsSizer;
      pcl::CheckBox       CorrectOnlyCheck;
      pcl::Label          NormLabel;
      pcl::ComboBox       NormCombo;
      pcl::HorizontalSizer NormSizer;

      // Additional Options
      pcl::SectionBar     AdditionalBar;
      pcl::Control        AdditionalSection;
      pcl::VerticalSizer  AdditionalSizer;
      pcl::NumericControl TileOverlap;
      pcl::CheckBox       ReflectPadCheck;
      pcl::Label          DeviceLabel;
      pcl::ComboBox       DeviceCombo;
      pcl::HorizontalSizer DeviceSizer;
      pcl::CheckBox       VerboseCheck;

      pcl::VerticalSizer  VSizer;
   };

   GUIData* GUI = nullptr;
};
