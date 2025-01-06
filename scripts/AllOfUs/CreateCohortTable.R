# Clear environment
rm(list = ls())
set.seed(13)

# Load necessary libraries
options(scipen = 100, digits = 3)
library('dplyr')
library('ggplot2')
library("data.table")
library("stringr")

# Check for command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Check if the required paths are provided (metadata and FAM file)
if (length(args) < 2) {
    stop("Error: Please provide the paths to the metadata directory and FAM file as command-line arguments.")
} else {
    metadata_path <- args[1]
    fam_file_path <- args[2]
}

# Load demographic data
demographic_file <- file.path(metadata_path, 'DemographicData.tsv')
demographics <- read.table(demographic_file, header = TRUE, sep = '\t')

# Select relevant columns from the demographic data
demographics <- demographics %>% select(person_id, race, ethnicity, date_of_birth, sex_at_birth)

# Check number of rows and first few rows
cat("Number of rows in demographics: ", nrow(demographics), "\n")
head(demographics)

# Create the Age column based on the year of birth
demographics$DateOfBirthYear <- as.numeric(gsub("-.*", "", demographics$date_of_birth))
demographics$Age <- as.numeric(gsub("-.*", "", as.character(Sys.Date()))) - demographics$DateOfBirthYear

# Load the FAM file to get genetic data
fam_file <- as.data.frame(fread(fam_file_path))

# Check number of rows and first few rows of FAM file
cat("Number of rows in FAM file: ", nrow(fam_file), "\n")
head(fam_file)

# Merge the demographic and genetic data
demographics <- merge(demographics, fam_file, by.x = "person_id", by.y = "V2")

# Process race and ethnicity for SelfReportedRaceEthnicity column
# Handle "No information" cases for certain race responses
demographics$SelfReportedRaceEthnicity <- ifelse(demographics$race %in% c("None Indicated", "I prefer not to answer", 
                                                                         "None of these", "PMI: Skip"), 
                                                 "No information", demographics$race)

# Count the distribution after initial labeling
cat("Initial count of SelfReportedRaceEthnicity:\n")
print(count(demographics, SelfReportedRaceEthnicity))

# Hispanic/Latino ethnicity label adjustment
demographics$SelfReportedRaceEthnicity <- ifelse(demographics$ethnicity == "Hispanic or Latino" & 
                                                 demographics$SelfReportedRaceEthnicity == "No information", 
                                                 "Hispanic or Latino", 
                                                 demographics$SelfReportedRaceEthnicity)

# Count the distribution after Hispanic/Latino label
cat("Count after Hispanic/Latino ethnicity label adjustment:\n")
print(count(demographics, SelfReportedRaceEthnicity))

# Handle cases where race is not "No information" but ethnicity is Hispanic/Latino
demographics$SelfReportedRaceEthnicity <- ifelse(demographics$ethnicity == "Hispanic or Latino" & 
                                                 demographics$SelfReportedRaceEthnicity != "Hispanic or Latino", 
                                                 "More than one population", 
                                                 demographics$SelfReportedRaceEthnicity)

# Final count of SelfReportedRaceEthnicity
cat("Final count of SelfReportedRaceEthnicity:\n")
print(count(demographics, SelfReportedRaceEthnicity))

# Get summary statistics for sex and ethnicity
cat("Counts by sex_at_birth:\n")
print(dplyr::count(demographics, sex_at_birth))

cat("Counts by SelfReportedRaceEthnicity and sex_at_birth:\n")
print(dplyr::count(demographics, SelfReportedRaceEthnicity, sex_at_birth))

# Age summary statistics
cat("Age mean and standard deviation:\n")
print(demographics %>% summarise_at(vars(Age), list(mean, sd)))

cat("Age mean and standard deviation by SelfReportedRaceEthnicity:\n")
print(demographics %>% group_by(SelfReportedRaceEthnicity) %>% summarise_at(vars(Age), list(mean, sd)))

# Write the processed demographic data to a new file for downstream use
output_file <- file.path(metadata_path, "ProcessedSIRE_V7.tsv")
write.table(demographics %>% select(person_id, SelfReportedRaceEthnicity), 
            file = output_file, sep = "\t", quote = FALSE, row.names = FALSE)

cat("Processed data written to:", output_file, "\n")