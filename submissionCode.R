#Example Submission
M = matrix(0, nrow=20000, ncol=2)

#first column: ID
M[,1] = 20001:40000

#2nd column: prediction.
M[,2] = sample(0:6, 20000, replace=TRUE)

#write csv
write.table(M, "submissionExample.csv",sep=",", row.names = FALSE, col.names = FALSE)
