###############################################################################
# STA211 - Analyse : Arbres de r?gression/classification & M?ta-algorithmes
# Pr?par? pour le samedi 6 juin 2026
# Inspir? des cours de V. Audigier & N. Niang
###############################################################################

# ---- 1. Pr?paration ----
# install.packages(c("rpart", "rpart.plot", "randomForest", "ipred", "ROCR", "ada", "caret"))

library(rpart)
library(rpart.plot)
library(randomForest)
library(ipred)
library(ROCR)
library(ada)

set.seed(235)
data("GermanCredit", package = "caret")
don <- GermanCredit
head(don)

# ---- 2. S?paration apprentissage/test ----
id  <- sample(seq(nrow(don)), size = ceiling(nrow(don) * 2 / 3))
ech <- don[ id, ]
test <- don[-id, ]

# ---- 3. Arbre CART ----
cat("===== ARBRE CART =====\n")

arbre <- rpart(Class ~ ., data = ech)
printcp(arbre)
plotcp(arbre)

# ?lagage
cp_opt <- arbre$cptable[which.min(arbre$cptable[, "xerror"]), "CP"]
arbre_prune <- prune(arbre, cp = cp_opt)

# Pr?diction
pred_proba <- predict(arbre_prune, newdata = test, type = "prob")[, "Good"]
pred_classe <- predict(arbre_prune, newdata = test, type = "class")

# Performance
pred_obj <- prediction(pred_proba, test$Class)
auc_cart <- performance(pred_obj, "auc")@y.values[[1]]
err_cart <- mean(pred_classe != test$Class)
cat("AUC CART:", round(auc_cart, 4), "\n")
cat("Erreur CART:", round(err_cart, 4), "\n")

# ---- 4. Bagging avec ipred ----
cat("\n===== BAGGING =====\n")

bag <- bagging(Class ~ .,
               data    = ech,
               nbagg   = 200,
               coob    = TRUE,
               control = rpart.control(minbucket = 5))

# Agr?gation des probabilit?s
pred_bag_prob <- predict(bag, newdata = test, type = "prob", aggregation = "average")
auc_bag <- performance(prediction(pred_bag_prob[, "Good"], test$Class), "auc")@y.values[[1]]

# Agr?gation par vote majoritaire
pred_bag_maj <- predict(bag, newdata = test, type = "prob", aggregation = "majority")
auc_bag_maj <- performance(prediction(pred_bag_maj[, "Good"], test$Class), "auc")@y.values[[1]]

cat("AUC Bagging (prob):", round(auc_bag, 4), "\n")
cat("AUC Bagging (vote):", round(auc_bag_maj, 4), "\n")
cat("Erreur OOB:", round(bag$err, 4), "\n")

# ---- 5. For?ts al?atoires ----
cat("\n===== FOR?TS AL?ATOIRES =====\n")

rf <- randomForest(Class ~ ., data = ech, ntree = 200, importance = TRUE)
pred_rf <- predict(rf, newdata = test, type = "prob")[, "Good"]
auc_rf  <- performance(prediction(pred_rf, test$Class), "auc")@y.values[[1]]
cat("AUC Random Forest:", round(auc_rf, 4), "\n")

# Importance des variables
cat("\nImportance des variables (MeanDecreaseGini):\n")
print(importance(rf, type = 2))
varImpPlot(rf, type = 2)

# ---- 6. Boosting avec ada ----
cat("\n===== BOOSTING (ada) =====\n")

boost <- ada(Class ~ ., data = ech, iter = 500,
             control = rpart.control(cp = -1, maxdepth = 1),
             nu = 1)
pred_boost <- predict(boost, newdata = test, type = "prob")[, 2]
auc_boost  <- performance(prediction(pred_boost, test$Class), "auc")@y.values[[1]]
cat("AUC Boosting (stumps):", round(auc_boost, 4), "\n")

# ---- 7. Synth?se des r?sultats ----
cat("\n===== SYNTH?SE =====\n")
resultats <- data.frame(
  Methode       = c("CART ?lagu?", "Bagging", "For?t al?atoire", "Boosting"),
  AUC           = round(c(auc_cart, auc_bag, auc_rf, auc_boost), 4),
  Erreur_test   = round(c(err_cart,
                          mean(predict(bag, test, type = "class", aggregation = "average") != test$Class),
                          mean(predict(rf,  test) != test$Class),
                          mean(predict(boost, test) != test$Class)), 4)
)
print(resultats)

###############################################################################
