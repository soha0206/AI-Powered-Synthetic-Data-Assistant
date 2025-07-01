CREATE TABLE dataset (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    "year" INTEGER,
    "industry_aggregation_nzsioc" TEXT,
    "industry_code_nzsioc" TEXT,
    "industry_name_nzsioc" TEXT,
    "units" TEXT,
    "variable_code" TEXT,
    "variable_name" TEXT,
    "variable_category" TEXT,
    "value" TEXT,
    "industry_code_anzsic06" TEXT
);