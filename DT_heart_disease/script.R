set.seed(123456) # for reproducibility

#Load the dataset and print summary information
data = read.table("heart.txt", header=TRUE, dec=".")
data$slope = factor(data$slope)
summary(data)

#Randomly split the data into training and validation sets
split <- sample(c(1, 2), size = nrow(data), replace = TRUE, prob = c(.7, .3))
data.train = data[split == 1, ]
data.test = data[split == 2, ]

#Fit an unconstrained decision tree
library(rpart)
settings = rpart.control(
  minsplit = 2,                     #The minimum number of samples required to split an internal node
	minbucket = 1,                    #The minimum number of samples for a leaf node
	cp=0                              #At the moment we don't want to prune the tree, 
                                    #so any increment in the complexity parameter is accepted
)

tree = rpart(
	num ~ .,                          #We want to model any relation between the outcome variable and all the features
	data=data.train,                  #The actual training data
	method="class",                   #We train a classification tree
	parms=list(split="gini"),         #We want to use the cross-entropy splitting criterion
	control=settings
)

#After training, we evaluate the model on both the training and the test sets
pred.train = predict(tree, data.train, type="class")
pred.test = predict(tree, data.test, type="class")

#The confusion matrix allows to visualize the type of errors made by the model
conf.train = table(data.train$num, pred.train)
conf.test = table(data.test$num, pred.test)

#Finally, we evaluate the accuracy
accuracy.train = sum(diag(conf.train))/sum(conf.train)
accuracy.test = sum(diag(conf.test))/sum(conf.test)

cat("Predictions of the grown Classification Tree for the training set:\n")
print(conf.train)
cat(c("Total accuracy: ", round(accuracy.train, 4) * 100, "%"))

cat("Predictions of the grown Classification Tree for the test set:\n")
print(conf.test)
cat(c("Total accuracy: ", round(accuracy.test, 4) * 100, "%"))

#Now, we prune the tree to reduce overfitting
#The functions plotcp and printcp show a graph and a table respectively
#of the cross-validated complexity parameter value (alpha)
#for each pruned sub-tree. The best pruned tree is the one that achieves
#the least cross-validated error (the y-axis of the graph or the xerror in the table)
plotcp(tree)
printcp(tree)

#Then we limit the tree at the complexity parameter corresponding to the lowest 
#cross-validation error
tree.pruned = prune(tree, cp=0.0176471)

#We evaluate the pruned tree
pred.train = predict(tree.pruned, data.train, type="class")
pred.test = predict(tree.pruned, data.test, type="class")

#The confusion matrix allows to visualize the type of errors made by the model
conf.train = table(data.train$num, pred.train)
conf.test = table(data.test$num, pred.test)

#Finally, we evaluate the accuracy
accuracy.train = sum(diag(conf.train))/sum(conf.train)
accuracy.test = sum(diag(conf.test))/sum(conf.test)

cat("Predictions of the pruned Classification Tree for the training set:\n")
print(conf.train)
cat(c("Total accuracy: ", round(accuracy.train, 4) * 100, "%"))

cat("Predictions of the pruned Classification Tree for the test set:\n")
print(conf.test)
cat(c("Total accuracy: ", round(accuracy.test, 4) * 100, "%"))

#Rule Extraction
library(rattle)
asRules(tree.pruned, compact=TRUE)
