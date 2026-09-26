-- ============================================================
-- HR Analytics - SQL queries for reporting and EDA
-- Demonstrates joins, aggregations, window functions, CTEs
-- ============================================================

-- Q1: Attrition rate by department
-- ---------------------------------
SELECT
    Department,
    COUNT(*) AS total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS attritions,
    ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_pct
FROM hr_raw
GROUP BY Department
ORDER BY attrition_pct DESC;


-- Q2: Attrition by job role (top 10 roles by volume)
-- -------------------------------------------------
SELECT
    JobRole,
    COUNT(*) AS n,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS left_count,
    ROUND(AVG(MonthlyIncome), 0) AS avg_monthly_income
FROM hr_raw
GROUP BY JobRole
HAVING COUNT(*) >= 20
ORDER BY n DESC
LIMIT 10;


-- Q3: Average tenure and satisfaction by attrition status
-- ------------------------------------------------------
SELECT
    Attrition,
    ROUND(AVG(YearsAtCompany), 2) AS avg_years_at_company,
    ROUND(AVG(JobSatisfaction), 2) AS avg_job_satisfaction,
    ROUND(AVG(EnvironmentSatisfaction), 2) AS avg_env_satisfaction,
    ROUND(AVG(WorkLifeBalance), 2) AS avg_work_life_balance
FROM hr_raw
GROUP BY Attrition;


-- Q4: Income distribution by education and attrition (CTE)
-- -------------------------------------------------------
WITH income_by_edu AS (
    SELECT
        EducationField,
        Attrition,
        MonthlyIncome,
        NTILE(4) OVER (PARTITION BY Attrition ORDER BY MonthlyIncome) AS income_quartile
    FROM hr_raw
)
SELECT EducationField, Attrition, income_quartile,
       ROUND(AVG(MonthlyIncome), 0) AS avg_income,
       COUNT(*) AS n
FROM income_by_edu
GROUP BY EducationField, Attrition, income_quartile
ORDER BY EducationField, Attrition, income_quartile;


-- Q5: Overtime and travel impact on attrition
-- ------------------------------------------
SELECT
    OverTime,
    BusinessTravel,
    COUNT(*) AS total,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS attritions,
    ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS attrition_pct
FROM hr_raw
GROUP BY OverTime, BusinessTravel
ORDER BY attrition_pct DESC;


-- Q6: Years since last promotion vs attrition (window aggregate)
-- --------------------------------------------------------------
SELECT
    YearsSinceLastPromotion,
    Attrition,
    COUNT(*) AS employees,
    ROUND(AVG(YearsAtCompany) OVER (PARTITION BY Attrition), 2) AS avg_tenure_by_attrition
FROM hr_raw
GROUP BY YearsSinceLastPromotion, Attrition
ORDER BY YearsSinceLastPromotion, Attrition;
