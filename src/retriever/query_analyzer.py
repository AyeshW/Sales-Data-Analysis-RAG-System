from __future__ import annotations

import re
from typing import Any


class QueryAnalyzer:
	def analyze(self, query: str) -> dict:
		"""
		Analyze the input query to detect filters for vector store retrieval.
        Returns a dictionary of filters to apply to the vector store query.
		"""
		text = query.lower()

		filters: list[dict[str, Any]] = []

		doc_type_filter = self._detect_doc_types(text)
		if doc_type_filter:
			filters.append({"doc_type": {"$in": doc_type_filter}})

		year_filter = self._detect_year(text)
		if year_filter is not None:
			filters.append({"year": year_filter})

		quarter_filter = self._detect_quarter(text)
		month_filter = self._detect_month(text)
		season_filter = self._detect_season(text)

		# Conflict resolution: month overrides quarter and season
		if month_filter is not None:
			quarter_filter = None
			season_filter = None

		# Conflict resolution: explicit quarter overrides season-derived
		if quarter_filter is not None:
			season_filter = None

		if quarter_filter is not None:
			filters.append({"quarter": quarter_filter})
		if month_filter is not None:
			filters.append({"month": month_filter})
		if season_filter is not None:
			filters.append({"season": season_filter})

		category_filter = self._detect_category(text)
		if category_filter is not None:
			filters.append({"category": category_filter})

		region_filter = self._detect_region(text)
		if region_filter is not None:
			filters.append({"region": region_filter})

		segment_filter = self._detect_segment(text)
		if segment_filter is not None:
			filters.append({"segment": segment_filter})

		if not filters:
			return {}

		if len(filters) == 1:
			return filters[0]

		return {"$and": filters}

	def explain(self, query: str) -> str:
		detected: list[str] = []
		text = query.lower()

		doc_types = self._detect_doc_types(text)
		if doc_types:
			detected.append(f"- doc_type: {', '.join(doc_types)}")

		year = self._detect_year(text)
		if year is not None:
			detected.append(f"- year: {year}")

		quarter = self._detect_quarter(text)
		month = self._detect_month(text)
		season = self._detect_season(text)

		if month is not None:
			quarter = None
			season = None

		if quarter is not None:
			season = None

		if quarter is not None:
			detected.append(f"- quarter: {quarter}")
		if month is not None:
			detected.append(f"- month: {month}")
		if season is not None:
			detected.append(f"- season: {season}")

		category = self._detect_category(text)
		if category is not None:
			detected.append(f"- category: {category}")

		region = self._detect_region(text)
		if region is not None:
			detected.append(f"- region: {region}")

		segment = self._detect_segment(text)
		if segment is not None:
			detected.append(f"- segment: {segment}")

		if not detected:
			return "No filters detected. Searching all documents."

		return "Detected filters:\n" + "\n".join(detected)

	@staticmethod
	def _detect_doc_types(text: str) -> list[str]:
		doc_types: list[str] = []

		yearly_terms = ["yearly", "annual", "year-over-year", "yoy", "overall", "year", "annually"]
		if any(term in text for term in yearly_terms):
			doc_types.extend(
				[
					"yearly_summary",
					"yearly_category_summary",
					"regional_yearly_summary",
					"comparative_yearly",
					"seasonality_pattern_overall",
				]
			)

		monthly_terms = ["monthly", "month-over-month", "mom", "month", "months"]
		if any(term in text for term in monthly_terms):
			doc_types.extend(
				[
					"monthly_summary",
					"yearly_summary",
				]
			)
		
		quarterly_seasonal_terms = [
			"quarterly",
			"quarter-over-quarter",
			"qoq",
			"quarter",
			"q1",
			"q2",
			"q3",
			"q4",
			"winter",
			"summer",
			"fall",
			"spring",
			"autumn",
			"seasonality",
			"season",
		]
		if any(term in text for term in quarterly_seasonal_terms):
			doc_types.extend(
				[
					"quarterly_summary",
					"quarterly_region_summary",
					"comparative_yearly",
					"seasonality_pattern_overall",
					"seasonality_summary",
				]
			)

		comparative_terms = [
			"trend",
			"growth",
			"compare",
			"comparison",
			"impact",
			"effect",
			"change",
			"increase",
			"decrease",
			"decline",
			"rise",
			"fall",
			"highest",
			"lowest",
			"best",
			"worst",
			"top",
			"most",
			"least",
			"versus",
			"vs",
			"against",
			"over",
		]
		if any(term in text for term in comparative_terms):
			doc_types.extend(
				[
					"seasonality_pattern_overall",
					"comparative_yearly",
					"comparative_category",
					"comparative_regional",
					"comparative_segment",
					"comparative_discount_impact",
				]
			)

		category_terms = [
			"category", 
			"sub-category", 
			"sub category", 
			"product line", 
			"technology", 
			"furniture", 
			"office supplies"
		]
		if any(term in text for term in category_terms):
			doc_types.extend(
				[
					"category_summary",
					"subcategory_summary",
					"region_category_summary",
					"comparative_category",
				]
			)

		region_terms = ["region", "state", "city", "west", "east", "central", "south"]
		if any(term in text for term in region_terms):
			doc_types.extend(
				[
					"regional_summary",
					"comparative_regional",
				]
			)

		seen: set[str] = set()
		ordered = []
		for doc_type in doc_types:
			if doc_type not in seen:
				seen.add(doc_type)
				ordered.append(doc_type)
		return ordered

	@staticmethod
	def _detect_year(text: str) -> int | None:
		years = re.findall(r"\b(201[4-7])\b", text)
		unique_years = sorted({int(year) for year in years})
		if len(unique_years) == 1:
			return unique_years[0]
		return None

	@staticmethod
	def _detect_quarter(text: str) -> int | None:
		matches = set()
		for q in re.findall(r"\bq([1-4])\b", text, flags=re.IGNORECASE):
			matches.add(int(q))

		quarter_phrases = {
			"first quarter": 1,
			"second quarter": 2,
			"third quarter": 3,
			"fourth quarter": 4,
		}
		for phrase, quarter in quarter_phrases.items():
			if phrase in text:
				matches.add(quarter)

		if len(matches) == 1:
			return next(iter(matches))
		return None

	@staticmethod
	def _detect_month(text: str) -> int | None:
		month_map = {
			"january": 1,
			"jan": 1,
			"february": 2,
			"feb": 2,
			"march": 3,
			"mar": 3,
			"april": 4,
			"apr": 4,
			"may": 5,
			"june": 6,
			"jun": 6,
			"july": 7,
			"jul": 7,
			"august": 8,
			"aug": 8,
			"september": 9,
			"sep": 9,
			"october": 10,
			"oct": 10,
			"november": 11,
			"nov": 11,
			"december": 12,
			"dec": 12,
		}

		matches = set()
		for token, month in month_map.items():
			if re.search(rf"\b{re.escape(token)}\b", text):
				matches.add(month)

		if len(matches) == 1:
			return next(iter(matches))
		return None

	@staticmethod
	def _detect_season(text: str) -> str | None:
		season_map = {
			"winter": "winter",
			"spring": "spring",
			"summer": "summer",
			"fall": "fall",
			"autumn": "fall",
		}

		matches = set()
		for token, season in season_map.items():
			if re.search(rf"\b{re.escape(token)}\b", text):
				matches.add(season)

		if len(matches) == 1:
			return next(iter(matches))
		return None

	@staticmethod
	def _detect_category(text: str) -> str | None:
		detected = set()
		if re.search(r"\btechnology\b", text):
			detected.add("Technology")
		if re.search(r"\bfurniture\b", text):
			detected.add("Furniture")
		if re.search(r"\boffice\s+supplies\b", text):
			detected.add("Office Supplies")

		if len(detected) == 1:
			return next(iter(detected))
		return None

	@staticmethod
	def _detect_region(text: str) -> str | None:
		region_map = {
			"west": "West",
			"east": "East",
			"central": "Central",
			"south": "South",
		}
		detected = set()
		for token, region in region_map.items():
			if re.search(rf"\b{re.escape(token)}\b", text):
				detected.add(region)

		if len(detected) == 1:
			return next(iter(detected))
		return None

	@staticmethod
	def _detect_segment(text: str) -> str | None:
		detected = set()
		if re.search(r"\bconsumer\b", text):
			detected.add("Consumer")
		if re.search(r"\bcorporate\b", text):
			detected.add("Corporate")
		if re.search(r"\bhome\s+office\b", text):
			detected.add("Home Office")

		if len(detected) == 1:
			return next(iter(detected))
		return None
