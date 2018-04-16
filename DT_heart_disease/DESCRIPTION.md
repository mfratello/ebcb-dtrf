# Dataset Description
The data used in this example is a subset of the [Cleveland Database][1], widely used as a benchmark for machine learning models.
The dataset comprises 270 observations of 76 attributes, however, the most common usages are limited to the 13 features described later.

## Features Description
1. age: age in years 
2. sex: sex
3. cp: chest pain type
	A: typical angina
	B: atypical angina
	C: non-anginal pain
	D: asymptomatic 
4. trestbps: resting blood pressure (in mm Hg) 
5. chol: serum cholestoral in mg/dl 
6. fbs:  fasting blood sugar > 120 mg/dl
	A: True
	B: False
7. restecg: resting electrocardiographic results
	A: normal
	B: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
	C: showing probable or definite left ventricular hypertrophy by Estes' criteria 	
8. thalach: maximum heart rate achieved 
9. exang: exercise induced angina (yes/no)
10. oldpeak = ST depression induced by exercise relative to rest 
11. slope: the slope of the peak exercise ST segment
	1: upsloping
	2: flat
	3: downsloping 
12. ca: number of major vessels (0-3) colored by flourosopy 
14. num: diagnosis of heart disease (angiographic disease status)
	absence: < 50% diameter narrowing
	presence: > 50% diameter narrowing

[1]: https://archive.ics.uci.edu/ml/datasets/Heart+Disease

# Setup
The example has been developed and tested using R 3.1.3 and RStudio 1.0.143.
See the RStudio [website][2] for installation instructions.

## Dependencies installation
To run the code and reproduce the results the `rpart` and `rattle` packages need to be installed:

```R
install.packages("rpart")
install.packages("rattle")
```

[2]: https://www.rstudio.com/
