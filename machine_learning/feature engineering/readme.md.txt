*********THE TYPES OF VARIOUS TECHNIQUES THAT CAN BE APPLIED DURINF EDA AND FE**************

FEATURE SCALING
	min-max scaler
	standard scaler
	robust scaler
	max abs scaler
	power transformer scaler
	quantile transformer scaler
	unit vector scaler
	
CATEGORICAL ENCODING:
	one-hot encoding (nominal data) - red, yellow, green, pink
		1. OneHotEncoding using Pandas
		2. K-1 OneHotEncoding
		3. OneHotEncoding using Sklearn
		4. OneHotEncoding with Top Categories (keeping other less frequent categories as OTHER)
	ordinal encoding (ordinal data) - PHD>PG>UG>HSC>SSC
	label encoding (for target variable)
	Hashing encoding
	mean encoding
	Target guided ordinal encoding
	Frequency encoding
	Helmert encoding
	Weight of evidence encoding
	Probability ratio encoding
	Leave one out encoding
	Backward differing encoding
	
MISSING VALUES IMPUTATION:
	Global constant
	Central tendency:
		mean
		median
		mode
	Central tendency for each class group
	knn imputation
	Prediction model imputation

Handling outliers:
	Detecting outliers:
		z-score
		IQR
		Boxplot
		Scatter plots
	Drop them
	Cap them

FEATURE TRANSFORMATION:
	Guassian transformation
	sklearn: FUNCTION transformer
		Log transformation
		Exponential transformation
		Reciprocol transformation
		square root transformation
		cube root transformation
	sklearn: POWER transformer
		yeo-johnson transformer
		Box cox transformation
	sklearn: QUALTILE transformer
		.
		.
	Left skewed data
	Right skewed data
	Q-Q plots

	

FEATURE EXTRACTION:
	constructing features

FEATURE SELECTION
	Correlation
	VIF
	Univariate selection
	PCA
	Recursive feature elimination
	Information Gain
	Extra trees classifier: model.feature_importances
	chi-sq test, t-test, ANOVA, one sample proportion test

IMBALANCED DATASET
	Undersampling
	Oversampling
	SMOTE TOMEK
	Easy ensemble classifier
	NearMiss
	Ensemble techniques

EXPLORATORY DATA ANALYSIS
	UNIVARIATE ANALYSIS
		Continuous variable:
			central tendencies
			min, max
			measure of dispersion
				Range
				Quantile
				IQR
				Variance
				Standard Deviation
				Skewness
				Kurtosis
			Visualization:
				histogram
				kde
				boxplot
		Categorical variable:
			frequency table
			count
			count %
			bar chart
	BIVARIATE ANALYSIS
		continuous-continuous
			scatter plots
			r (+1 to -1)
		continuous-categorical
			z-test
			t-tets
			ANOVA
		categorical-categorical
			Two way table
			Stacked column chart
			Chi-sq test
	MULTIVARIATE ANALYSIS

CROSS VALIDATION:
	Leave one out
	K-fold
	Stratified
	Time Series CV
	
		