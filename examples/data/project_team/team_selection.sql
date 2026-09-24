PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS assignments;
DROP TABLE IF EXISTS project_experience;
DROP TABLE IF EXISTS availability;
DROP TABLE IF EXISTS person_skills;
DROP TABLE IF EXISTS skills;
DROP TABLE IF EXISTS people;

CREATE TABLE people (
    person_id INTEGER PRIMARY KEY,
    full_name TEXT NOT NULL UNIQUE,
    role TEXT NOT NULL,
    hourly_rate_usd INTEGER NOT NULL CHECK (hourly_rate_usd > 0),
    location TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1))
);
CREATE TABLE skills (skill_id INTEGER PRIMARY KEY, skill_name TEXT NOT NULL UNIQUE);
CREATE TABLE person_skills (
    person_id INTEGER NOT NULL REFERENCES people(person_id),
    skill_id INTEGER NOT NULL REFERENCES skills(skill_id),
    proficiency INTEGER NOT NULL CHECK (proficiency BETWEEN 1 AND 5),
    years_experience INTEGER NOT NULL CHECK (years_experience >= 0),
    last_used_year INTEGER NOT NULL,
    PRIMARY KEY (person_id, skill_id)
);
CREATE TABLE availability (
    person_id INTEGER PRIMARY KEY REFERENCES people(person_id),
    available_from TEXT NOT NULL,
    available_to TEXT NOT NULL,
    hours_per_week INTEGER NOT NULL CHECK (hours_per_week BETWEEN 0 AND 40)
);
CREATE TABLE project_experience (
    experience_id INTEGER PRIMARY KEY,
    person_id INTEGER NOT NULL REFERENCES people(person_id),
    project_name TEXT NOT NULL,
    project_type TEXT NOT NULL,
    outcome TEXT NOT NULL
);
CREATE TABLE assignments (
    assignment_id INTEGER PRIMARY KEY,
    person_id INTEGER NOT NULL REFERENCES people(person_id),
    project_name TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    hours_per_week INTEGER NOT NULL CHECK (hours_per_week BETWEEN 1 AND 40)
);

INSERT INTO skills VALUES
    (1, 'Python'), (2, 'SQL'), (3, 'Communication'),
    (4, 'Project coordination'), (5, 'Data analysis'),
    (6, 'Testing'), (7, 'Documentation');

WITH RECURSIVE candidate(person_id) AS (
    SELECT 1 UNION ALL SELECT person_id + 1 FROM candidate WHERE person_id < 50
)
INSERT INTO people (person_id, full_name, role, hourly_rate_usd, location)
SELECT person_id, 'Candidate ' || printf('%02d', person_id),
    CASE person_id % 5 WHEN 0 THEN 'Project coordinator' WHEN 1 THEN 'Software developer'
         WHEN 2 THEN 'Data analyst' WHEN 3 THEN 'Quality analyst' ELSE 'Technical writer' END,
    48 + ((person_id * 7) % 38),
    CASE person_id % 4 WHEN 0 THEN 'Berlin' WHEN 1 THEN 'Munich' WHEN 2 THEN 'Remote' ELSE 'Hamburg' END
FROM candidate;

UPDATE people SET full_name='Alice Chen', role='Software developer', hourly_rate_usd=60, location='Remote' WHERE person_id=1;
UPDATE people SET full_name='Ben Ortiz', role='Data analyst', hourly_rate_usd=55, location='Berlin' WHERE person_id=2;
UPDATE people SET full_name='Cara Singh', role='Project coordinator', hourly_rate_usd=65, location='Remote' WHERE person_id=3;
UPDATE people SET full_name='Daniel Weber', role='Senior developer', hourly_rate_usd=82, location='Munich' WHERE person_id=4;
UPDATE people SET full_name='Eva Novak', role='Data analyst', hourly_rate_usd=58, location='Hamburg' WHERE person_id=5;
UPDATE people SET full_name='Farid Khan', role='Quality analyst', hourly_rate_usd=52, location='Berlin' WHERE person_id=6;
UPDATE people SET full_name='Greta Müller', role='Project coordinator', hourly_rate_usd=70, location='Munich' WHERE person_id=7;
UPDATE people SET full_name='Hugo Martin', role='Software developer', hourly_rate_usd=62, location='Remote' WHERE person_id=8;

WITH RECURSIVE candidate(person_id) AS (
    SELECT 1 UNION ALL SELECT person_id + 1 FROM candidate WHERE person_id < 50
), generated AS (
    SELECT person_id, 1 skill_id, 1 + (person_id % 3) proficiency, 1 + (person_id % 8) years FROM candidate
    UNION ALL SELECT person_id, 2, 1 + ((person_id * 3) % 3), 1 + ((person_id * 2) % 8) FROM candidate
    UNION ALL SELECT person_id, 3, 1 + ((person_id * 5) % 3), 1 + ((person_id * 4) % 8) FROM candidate
    UNION ALL SELECT person_id, 4, 1 + ((person_id * 7) % 3), person_id % 7 FROM candidate
    UNION ALL SELECT person_id, 5 + (person_id % 3), 2 + ((person_id * 11) % 4), 1 + (person_id % 6) FROM candidate
)
INSERT INTO person_skills
SELECT person_id, skill_id, proficiency, years, 2026 - (person_id % 3) FROM generated;

UPDATE person_skills SET proficiency=5, years_experience=7, last_used_year=2026 WHERE person_id=1 AND skill_id=1;
UPDATE person_skills SET proficiency=4, years_experience=5, last_used_year=2026 WHERE person_id=1 AND skill_id IN (2,3);
UPDATE person_skills SET proficiency=5, years_experience=8, last_used_year=2026 WHERE person_id=2 AND skill_id=2;
UPDATE person_skills SET proficiency=4, years_experience=5, last_used_year=2026 WHERE person_id=2 AND skill_id IN (3,4);
UPDATE person_skills SET proficiency=4, years_experience=5, last_used_year=2026 WHERE person_id=3 AND skill_id=1;
UPDATE person_skills SET proficiency=5, years_experience=8, last_used_year=2026 WHERE person_id=3 AND skill_id=3;
UPDATE person_skills SET proficiency=5, years_experience=7, last_used_year=2026 WHERE person_id=3 AND skill_id=4;
UPDATE person_skills SET proficiency=5, years_experience=10, last_used_year=2026 WHERE person_id=4 AND skill_id=1;
UPDATE person_skills SET proficiency=5, years_experience=9, last_used_year=2026 WHERE person_id=5 AND skill_id=2;
UPDATE person_skills SET proficiency=4, years_experience=6, last_used_year=2026 WHERE person_id=6 AND skill_id=3;
UPDATE person_skills SET proficiency=5, years_experience=8, last_used_year=2026 WHERE person_id=7 AND skill_id=4;
UPDATE person_skills SET proficiency=4, years_experience=6, last_used_year=2026 WHERE person_id=8 AND skill_id=1;

WITH RECURSIVE candidate(person_id) AS (
    SELECT 1 UNION ALL SELECT person_id + 1 FROM candidate WHERE person_id < 50
)
INSERT INTO availability
SELECT person_id,
    CASE WHEN person_id % 9=0 THEN '2026-10-15' ELSE '2026-10-01' END,
    CASE WHEN person_id % 11=0 THEN '2026-10-20' ELSE '2026-11-30' END,
    16 + ((person_id * 4) % 25)
FROM candidate;

UPDATE availability SET hours_per_week=32 WHERE person_id=1;
UPDATE availability SET hours_per_week=28 WHERE person_id=2;
UPDATE availability SET hours_per_week=30 WHERE person_id=3;
UPDATE availability SET hours_per_week=12 WHERE person_id=4;
UPDATE availability SET hours_per_week=30, available_from='2026-10-20' WHERE person_id=5;
UPDATE availability SET hours_per_week=26 WHERE person_id=6;
UPDATE availability SET hours_per_week=24 WHERE person_id=7;
UPDATE availability SET hours_per_week=28 WHERE person_id=8;

INSERT INTO project_experience VALUES
    (1,1,'Customer Data Cleanup','Data migration','Delivered two days early'),
    (2,1,'Internal Reporting Tool','Software delivery','Passed acceptance testing'),
    (3,2,'Sales Metrics Refresh','Data analysis','Reduced report errors by 30 percent'),
    (4,2,'Inventory Reconciliation','Data migration','Resolved all priority discrepancies'),
    (5,3,'Support Dashboard','Software delivery','Coordinated four-person team on schedule'),
    (6,3,'Records Consolidation','Data migration','Completed without service interruption'),
    (7,4,'Billing Rewrite','Software delivery','Strong code quality; schedule slipped one week'),
    (8,5,'Warehouse Forecast','Data analysis','Accurate results delivered after deadline'),
    (9,6,'Regression Test Refresh','Quality assurance','Expanded coverage from 65 to 88 percent'),
    (10,7,'Office Relocation','Coordination','Delivered within budget'),
    (11,8,'API Monitoring','Software delivery','Reduced incident detection time'),
    (12,8,'Data Export Utility','Data migration','Passed acceptance testing');

WITH RECURSIVE candidate(person_id) AS (
    SELECT 9 UNION ALL SELECT person_id + 1 FROM candidate WHERE person_id < 50
)
INSERT INTO project_experience
SELECT 100+person_id, person_id, 'Department Project ' || printf('%02d',person_id),
    CASE person_id%3 WHEN 0 THEN 'Data migration' WHEN 1 THEN 'Software delivery' ELSE 'Coordination' END,
    CASE person_id%4 WHEN 0 THEN 'Delivered on schedule' WHEN 1 THEN 'Delivered one week late'
         WHEN 2 THEN 'Met quality target' ELSE 'Required minor rework' END
FROM candidate;

INSERT INTO assignments VALUES
    (1,6,'Release Verification','2026-09-15','2026-10-31',12),
    (2,7,'Office Systems Upgrade','2026-10-01','2026-11-15',18),
    (3,8,'API Maintenance','2026-09-01','2026-12-01',10),
    (4,12,'Quarterly Reporting','2026-10-01','2026-10-31',16),
    (5,18,'Inventory Audit','2026-09-20','2026-11-05',20),
    (6,24,'Website Refresh','2026-10-10','2026-12-15',12),
    (7,30,'Customer Survey','2026-10-01','2026-10-31',14),
    (8,36,'Access Review','2026-09-01','2026-11-30',20),
    (9,42,'Archive Migration','2026-10-01','2026-12-20',18),
    (10,48,'Training Rollout','2026-09-15','2026-11-15',16);

CREATE INDEX idx_person_skills_skill ON person_skills(skill_id, proficiency);
CREATE INDEX idx_experience_person ON project_experience(person_id);
CREATE INDEX idx_assignments_person_dates ON assignments(person_id,start_date,end_date);
