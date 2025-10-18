#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.folder_normalizer import CarFolderNormalizer

def test_consolidations():
    """
    Test model consolidations by showing before/after examples
    """

    # Test cases covering our consolidation batches
    test_cases = [
        # Batch 1: Door variations
        "ALFA_ROMEO_147_3_Doors_2003",
        "ALFA_ROMEO_147_5_Doors_2003",
        "BMW_1_Series_3_doors_E81_2007",
        "BMW_1_Series_5_doors_2007",
        "AUDI_A1_Sportback_5_doors_2019",
        "AUDI_A3_Hatchback_3_doors_2016",
        "HYUNDAI_Accent_3_Doors_2015",
        "HYUNDAI_Accent_4_Doors_2015",
        "HONDA_Accord_3_Doors_2008",
        "OPEL_Astra_3_Doors_2010",
        "CHEVROLET_AveoKalos_3_Doors_2008",
        "CITROEN_C1_3_Doors_2012",
        "HONDA_Civic_3_Doors_2016",
        "FORD_Fiesta_3_doors_2018",
        "FORD_Focus_3_Doors_2015",

        # Batch 3: Trim/Engine variants
        "AUDI_A4_DTM_Edition_2004",
        "AUDI_A1_Quattro_2019",
        "AUDI_A3_Sportback_etron_2020",
        "AUDI_A8_D2_1999",
        "AUDI_A8_L_2018",
        "FIAT_500C_Abarth_2015",
        "FIAT_595_Abarth_2018",
        "FIAT_500_esseesse_2010",
        "BMW_M3_CS_2018",
        "BMW_M4_CSL_2022",
        "BMW_M5_CS_2021",
        "DODGE_Challenger_SRT_2015",
        "DODGE_Charger_SRT8_2012",
        "FORD_Mustang_Shelby_GT500_2020",

        # Batch 4: BMW generations
        "BMW_1_Series_E87_2007",
        "BMW_1_Series_F20_2015",
        "BMW_1_Series_Cabriolet_E88_2008",
        "BMW_1_Series_Coupe_E82_2010",
        "BMW_3_Series_E46_2002",
        "BMW_3_Series_E90_2008",
        "BMW_3_Series_F30_2015",
        "BMW_3_Series_Cabriolet_E93_2010",
        "BMW_3_Series_Touring_E91_2008",
        "BMW_5_Series_E60_2005",
        "BMW_5_Series_F10_2012",
        "BMW_7_Series_E38_2001",
        "BMW_7_Series_F0102_2015"
    ]

    # Create normalizer (dummy path since we're not reading files)
    normalizer = CarFolderNormalizer("dummy")

    print("🔍 Testing Model Consolidations")
    print("=" * 90)
    print(f"{'Original Directory':<45} {'Make':<12} {'Model':<25} {'Year'}")
    print("=" * 90)

    consolidation_examples = {}

    for directory_name in test_cases:
        make, model, year = normalizer.extract_car_info(directory_name)
        print(f"{directory_name:<45} {make:<12} {model:<25} {year}")

        # Track consolidations
        original_model = directory_name.split('_')[1:-1]  # Extract model part
        original_model_str = '_'.join(original_model)
        if original_model_str != model:
            if model not in consolidation_examples:
                consolidation_examples[model] = []
            consolidation_examples[model].append(original_model_str)

    print("\n" + "=" * 90)
    print("📊 CONSOLIDATION SUMMARY")
    print("=" * 90)

    if consolidation_examples:
        for consolidated_model, original_variants in consolidation_examples.items():
            print(f"✅ {consolidated_model}:")
            for variant in set(original_variants):
                print(f"   ← {variant}")
            print()
    else:
        print("❌ No consolidations detected - check the mapping!")

    print("✅ Consolidation test completed!")
    print("💡 If results look good, proceed with cache creation")

if __name__ == "__main__":
    test_consolidations()