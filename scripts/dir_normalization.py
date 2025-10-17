"""
Script for normalizing directory names in the `car-models` dataset
Uses CarFolderNormalizer class
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

print(f"📍 Skripta: {SCRIPT_DIR}")
print(f"📍 Koren projekta: {PROJECT_ROOT}")
print(f"📍 Dataset: {DATASET_PATH}")

sys.path.insert(0, SRC_DIR)

try:
    from utils.folder_normalizer import CarFolderNormalizer
except ImportError as e:
    print(f"❌ Error while importing: {e}")
    print(f"ℹ️  Check if {SRC_DIR} exists and contains utils/folder_normalizer.py")
    sys.exit(1)


def main():
    print("\n🔧")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset path doesn't exist': {DATASET_PATH}")
        print("ℹ️  Create 'data/raw' directory and place car directories there")
        return

    os.makedirs(REPORTS_DIR, exist_ok=True)

    normalizer = CarFolderNormalizer(DATASET_PATH)

    items = os.listdir(DATASET_PATH)
    folders = [item for item in items if os.path.isdir(os.path.join(DATASET_PATH, item))]

    print(f"📁 Found {len(folders)} directories in dataset")

    if not folders:
        print("❌ No directories to work with!")
        return

    print("\n🧪 DRY RUN - What would change...")
    dry_run_results = normalizer.rename_directories(dry_run=True)

    changes = [r for r in dry_run_results if r[2] != "NO_CHANGE"]
    no_changes = [r for r in dry_run_results if r[2] == "NO_CHANGE"]

    if not changes:
        print("✅ All directories have the correct format!")
        return

    print(f"🔄 Found {len(changes)} directories for renaming")
    print(f"✅ Already formatted: {len(no_changes)}")

    print("\n📝 First 10 changes:")
    for old, new, status in changes[:10]:
        print(f"   {old}")
        print(f"   → {new}")
        print()

    if len(changes) > 10:
        print(f"   ... and {len(changes) - 10} more directories")

    stats = {}
    for _, _, status in dry_run_results:
        stats[status] = stats.get(status, 0) + 1

    print(f"\n📊 STATS (Dry Run):")
    for status, count in stats.items():
        print(f"   {status}: {count}")

    print(f"\n⚠️  READY TO RENAME")
    response = input(f"❓ Rename {len(changes)} directories? (y/N): ")

    if response.lower() == 'y':
        print("\n⚡ Renaming...")

        results = normalizer.rename_directories(dry_run=False)

        report_file = os.path.join(REPORTS_DIR, "folder_renaming_report.txt")
        normalizer.save_mapping_report(results, report_file)

        success_count = len([r for r in results if "RENAMED" in r[2]])
        error_count = len([r for r in results if "ERROR" in r[2] or "SKIP" in r[2]])
        no_change_count = len([r for r in results if "NO_CHANGE" in r[2]])

        print(f"\n🎉 RENAMING COMPLETED!")
        print(f"   ✅ Successfully renamed: {success_count}")
        print(f"   ❌ Errors/Skipped: {error_count}")
        print(f"   ✅ Already in correct format: {no_change_count}")
        print(f"   📊 Total folders: {len(results)}")
        print(f"   📄 Report saved: {report_file}")

        errors = [r for r in results if "ERROR" in r[2] or "SKIP" in r[2]]
        if errors:
            print(f"\n⚠️  Errors/Skipped ({len(errors)}):")
            for old, new, status in errors[:5]:
                print(f"   {status}: {old}")
            if len(errors) > 5:
                print(f"   ... and {len(errors) - 5} more")

        else:
            print("❌ Renaming cancelled")

            report_file = os.path.join(REPORTS_DIR, "folder_renaming_dry_run.txt")
            normalizer.save_mapping_report(dry_run_results, report_file)
            print(f"📄 Dry run report saved: {report_file}")

if __name__ == "__main__":
    main()
