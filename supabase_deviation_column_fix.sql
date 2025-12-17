-- SQL to fix the deviation column type in Supabase
-- The deviation column should be DOUBLE PRECISION (float), not INTEGER
-- Run this in your Supabase SQL Editor

-- Change deviation column from INTEGER to DOUBLE PRECISION
ALTER TABLE zelestra_comments 
ALTER COLUMN deviation TYPE DOUBLE PRECISION USING deviation::DOUBLE PRECISION;

-- If the above doesn't work (column might not exist yet), use:
-- ALTER TABLE zelestra_comments 
-- ADD COLUMN IF NOT EXISTS deviation DOUBLE PRECISION;


