
install.packages("httr")
library(RCurl)

library(readr)
IMDB<- read_csv("https://raw.githubusercontent.com/bharat284/MyLearningData/master/IMDB_data.csv")[-2,]

head(IMDB)

IMDB_Genre<- subset(IMDB['Genre'])
IMDB_Genre
nrow(IMDB_Genre)

Genre_Unique <- unique(IMDB_Genre)
nrow(Genre_Unique)


typeof(Genre_Unique)

Genre_Unique<-as.data.frame(Genre_Unique)






str(Genre_Unique)



Sort_Genre<-Genre_Unique[order(Genre_Unique),]

head(Sort_Genre)




colnames(IMDB)

IMDB$newcol<-(IMDB$imdbRating - IMDB$imdbVotes)^2

head(IMDB)

new_IMDB<-write.csv(IMDB, "New_IMDB")
