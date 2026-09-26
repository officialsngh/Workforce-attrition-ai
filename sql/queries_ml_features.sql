-- ============================================================
-- ML feature set: one row per employee, numeric/categorical for modeling
-- Target: Attrition (Yes=1, No=0). Excludes constants and IDs.
-- ============================================================

SELECT
    CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END AS target,
    Age,
    CASE BusinessTravel
        WHEN 'Non-Travel' THEN 0 WHEN 'Travel_Rarely' THEN 1 WHEN 'Travel_Frequently' THEN 2
        ELSE -1 END AS BusinessTravel,
    DailyRate,
    Department,
    DistanceFromHome,
    Education,
    EducationField,
    EnvironmentSatisfaction,
    Gender,
    HourlyRate,
    JobInvolvement,
    JobLevel,
    JobRole,
    JobSatisfaction,
    MaritalStatus,
    MonthlyIncome,
    MonthlyRate,
    NumCompaniesWorked,
    CASE WHEN OverTime = 'Yes' THEN 1 ELSE 0 END AS OverTime,
    PercentSalaryHike,
    PerformanceRating,
    RelationshipSatisfaction,
    StockOptionLevel,
    TotalWorkingYears,
    TrainingTimesLastYear,
    WorkLifeBalance,
    YearsAtCompany,
    YearsInCurrentRole,
    YearsSinceLastPromotion,
    YearsWithCurrManager
FROM hr_raw
WHERE EmployeeCount IS NOT NULL;
