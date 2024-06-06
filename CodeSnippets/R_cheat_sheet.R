
# Comments: # is used to add comments.
# Assignment: <- or = is used for assignment.
# Function Call: function_name(arguments)
# Vectors: c() function is used to create vectors.
# Data Types: Numeric, Character, Logical, Factor, Date, etc.
# Indexing: [ ] is used to subset elements of vectors or data frames.
# Functions: Defined using function() keyword.

# Install Package: install.packages("package_name")
# Load Package: library(package_name)

# Read Data: read.csv(), read.table()
# Data Exploration: head(), summary(), str(), dim()
# Subset Data: subset(), [ ]
# Filter Data: filter() (dplyr package)
# Join Data: merge(), join() (dplyr package)
# Transform Data: mutate() (dplyr package)
# Aggregate Data: aggregate() (base), summarize() (dplyr package)

# Base Graphics: plot(), hist(), boxplot()
# ggplot2 Package: ggplot(), geom_point(), geom_bar()
# Save Workspace: save.image("file.RData")
# Load Workspace: load("file.RData")
# Help: help(function_name), ?function_name

if (condition) {
  # code block
} else {
  # code block
}

for (variable in sequence) {
  # code block
}

while (condition) {
  # code block
}

repeat {
  # code block
  if (condition) {
    break
  }
}

function_name <- function(arg1, arg2, ...) {
  # code block
  return(result)
}

function(arg1, arg2, ...) {
  # code block
  return(result)
}

# glimpse
install.packages("pillar")
library(tibble)
glimpse(mpg)

# help about any method or dataset
?methodName

# view dataset
view(mpg)

# filter dataset
mpg_filtered <- filter(mpg, cty >= 20)