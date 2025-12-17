-- SQL to add missing columns to zelestra_comments table in Supabase
-- Run this in your Supabase SQL Editor

-- Add threshold column (if missing)
ALTER TABLE zelestra_comments 
ADD COLUMN IF NOT EXISTS threshold DOUBLE PRECISION;

-- Verify the table structure (optional - run to see current schema)
-- SELECT column_name, data_type 
-- FROM information_schema.columns 
-- WHERE table_name = 'zelestra_comments';


