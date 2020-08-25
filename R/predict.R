library(data.table)
library(h2o)

source("R/load.R")

# use H2o with 4 cores
h2o.init(nthreads = 4)

#load our data
data <- load()

# make sure we are predicting binomial
data$DEATH_EVENT <- as.factor(data$DEATH_EVENT)

# make train test split
smp_size <- floor(0.75 * nrow(data))
# set.seed(123)
train_index <- sample(seq_len(nrow(data)), size = smp_size)

train <- as.h2o(data[train_index, ])
test <- as.h2o(data[-train_index, ])

# setup predictors and response
y <- "DEATH_EVENT"
x <- setdiff(names(train), y)

# do AutoML magic
aml <- h2o.automl(x = x, y = y,
                  training_frame = train,
                  max_models = 20,
                  max_runtime_secs = 600,
                  seed = 1)

# pick best performing model and predict 
lb <- aml@leaderboard
pred <- h2o.predict(aml@leader, test)

# transform and combine predictions with test data
test <- as.data.table(test)
predictions <- as.vector(pred$predict)

test <- test[, pred := predictions][, list(DEATH_EVENT, pred)]

#calculate accuracy
accuracy <- nrow(test[DEATH_EVENT == pred]) / nrow(test)
print(paste0("Accuracy: ", accuracy))

