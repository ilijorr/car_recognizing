from pathlib import Path
from typing import List, Tuple

from utils.car_brands import KNOWN_BRANDS, NORMALIZED_BRANDS


class CarFolderNormalizer:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.KNOWN_BRANDS = KNOWN_BRANDS
        self.NORMALIZED_BRANDS = NORMALIZED_BRANDS

    def extract_car_info(self, directory_name: str) -> Tuple[str, str, str]:
        """Returns the information about a directory containing car images

            Parameters:
                directory_name (str): name of the directory to examine

            Returns:
                make (str): car manufacturer name
                model (str): model name
                year(str): car production period, in the format
                    "firstYear-lastYear" or
                    "year", depending on the available data
        """

        year_raw = directory_name.split('_')[-1]
        make_model_raw = directory_name.removesuffix('_' + year_raw)

        year = self._parse_year(year_raw)
        make, model = self._parse_make_model(make_model_raw)

        return make, model, year

    def normalize_directory_name(self, directory_name: str) -> str:
        """Renames the directory to follow the "MAKE_MODEL_GENERATION" format

            Parameters:
                directory_name (str): name of the directory to normalize

            Returns:
                normalized_name (str)
        """

        directory_name = directory_name.strip().replace('__', '_')

        if len(directory_name) >= 4 and directory_name[:4].isdigit():
            if len(directory_name) >= 5 and directory_name[4] != '_':
                directory_name = directory_name[:4] + '_' + directory_name[4:]
            elif len(directory_name) == 4:
                directory_name = directory_name + '_'

        segments = directory_name.strip().split('_')

        segments = [seg for seg in segments if seg]

        if not segments:
            return "UNKNOWN_UNKNOWN_UNKNOWN"

        if segments[0].isdigit() and len(segments[0]) == 4:
            year = segments[0]
            make_model = '_'.join(segments[1:])
            return f"{make_model}_{year}"
        else:
            return directory_name

    def get_directory_mapping(self) -> List[Tuple[str, str]]:
        """Returns old names mapped to normalized names."""
        mapping = []

        for item in self.data_path.iterdir():
            if item.is_dir():
                old_name = item.name
                new_name = self.normalize_directory_name(old_name)
                mapping.append((old_name, new_name))

        return mapping

    def rename_directories(self, dry_run: bool = True) -> List[Tuple[str, str, str]]:
        """Renames the directories to normalized name

            Parameters:
                dry_run (bool): if True, only shows what would change
                                if False, actually renames the directories
        """
        mapping = self.get_directory_mapping()
        results = []

        for old_name, new_name in mapping:
            old_path = self.data_path / old_name
            new_path = self.data_path / new_name

            if old_name == new_name:
                results.append((old_name, new_name, "NO_CHANGE"))
                continue

            if dry_run:
                results.append((old_name, new_name, "WOULD_RENAME"))
                continue

            try:
                if new_path.exists():
                    results.append((old_name, new_name, "SKIP_EXISTS"))
                    continue
                old_path.rename(new_path)
                results.append((old_name, new_name, "RENAMED"))
            except Exception as e:
                results.append((old_name, new_name, f"ERROR: {e}"))

        return results

    def save_mapping_report(self,
                            results: List[Tuple[str, str, str]],
                            output_file: str):
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("DIRECTORY NORMALIZATION REPORT\n")
            f.write("=" * 80 + "\n")

            for old, new, status in results:
                f.write(f"{status:15} | {old} -> {new}\n")

            stats = {}
            for _, _, status in results:
                stats[status] = stats.get(status, 0) + 1

            f.write("\nSTATS:\n")
            for status, count in stats.items():
                f.write(f"{status}: {count}\n")

    def _parse_year(self, year_str: str) -> str:
        """Parse year and convert to decade ranges"""
        # Take first 3 digits and add '0s' for decade
        # Examples: "1968" -> "1960s", "2022Present" -> "2020s", "20002010" -> "2000s"
        try:
            first_three_digits = year_str[:3]
            if first_three_digits.isdigit():
                return first_three_digits + "0s"
            else:
                return "Unknown"
        except:
            return "Unknown"

    def _parse_make_model(self, full_name: str) -> Tuple[str, str]:
        """Returns the car make and model"""

        # Convert to uppercase for consistent matching
        full_name_upper = full_name.upper()

        for brand in sorted(self.KNOWN_BRANDS, key=len, reverse=True):
            if full_name_upper.startswith(brand + '_'):
                # Extract model (everything after brand and underscore)
                model = full_name[len(brand) + 1:]  # +1 for the underscore

                if brand in self.NORMALIZED_BRANDS:
                    return self.NORMALIZED_BRANDS[brand], model
                else:
                    return brand, model

        return "UNKNOWN", "UNKNOWN"
