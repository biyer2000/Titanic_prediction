# Titanic_prediction
Predicts whether a passenger would survive the sinking of the Titanic ship. It uses a decision tree with either an option for parameterization(such as maximum depth) or to go up to a dead end.
The output is binary with 1 indicating the passenger survived and 0 indicating that the passenger did not survive. 
Input data (training and testing) is represented as a csv file which contains various attritues such as sex, age, Fare, Social Class, Place of Departure as shown below:


  Pclass	Sex	Age	Fare	Embarked	relatives	IsAlone	
0	 3	     1	  1	 11	    0	        2	        0	        
1	 3	     0	 28	 7	    1	        0	        1	

Used the pandas dataframe to load the above data and median filled null values for rows that have incomplete data. 

Used cross validation with a k parameter to split the whole dataset into 1/k validation data and the rest training data. Aprroximately 88% training accuracy and 78% testing accuracy was achieved on average. 



