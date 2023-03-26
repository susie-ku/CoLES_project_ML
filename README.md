# CoLES_project_ML
Project is dedicated to studying the effect of pre-train data size, model size on model accuracy and NN saturation. We applied a self-supervised learning method for embedding discrete event sequences called COntrastive Learning for Event Sequences (CoLES).

In this work, we evaluate the performance of CoLES on several benchmark datasets for event sequence analysis. Our goal is to investigate the influence of pre-train data size and model size on model accuracy and NN saturation.

We operate on two datasets with bank clients transactions information. **Sber dataset** contains information about transactions of bank customers. About 27,000,000 million records in volume. Each entry describes one banking transaction. 

Training transactions **train.csv**, in which the date, amount, type and id of the client are known for each transaction;

Test transactions **test.csv** containing the same fields: client id is a unique client number, trans date is the date of the transaction (it is simply the day number in chronological order,starting from the given date), small group is a group of transactions that characterize the type of transaction (for example, grocery stores , clothes, gas stations, children’s goods, etc.), amount rur - the amount of the transaction (for anonymization, these amounts were transformed without losing the structure).

On the database of files, you can build various features that characterize age groups. The target variable for the training dataset is in the train **target.csv** file. It contains information about the Client and the label of the age group to which it belongs: client id – unique number of the Client (corresponds to client id from the transactions train.csv file), bins – age label. In the **test.csv** file, we need to predict the corresponding age group label for the specified client id.

We are also provided with an information file small group **description.csv**, which contains a breakdown of transaction types. Based on these data, it is necessary to carry out a multiclass classification (4 classes - from 0 to 3). The quality of the solution is calculated as the proportion of correctly guessed age labels for all test cases -accuracy.

The **Rosbank dataset** contains information on about 520,000,000 million records. A record is a description of a particular customer’s transaction.

We have with one training dataset **train.csv**, in which for each transaction the date, amount, transaction type and client id, date of commission, amount, target flag and target sum are known.

**test.csv** containing the same fields: cl id - unique client number, TRDATETIME - date of the transaction (number of the day in chronological order, starting from the specified date), trx category - category of transactions characterizing the type of transaction, amount - transaction amount (for anonymization, these amounts were transformed without loss of structure). Based on the file database, it is possible to determine whether the customer will remain a user of the bank or not. The target variable for the training dataset is in the **train.csv** file. In the **test.csv** file, we need to predict the churn rate for the specified data. Based on these data, it is necessary to carry out a binary classification.

The quality of the solution is calculated as the proportion of correctly guessed age labels for all test cases - AUC ROC.
