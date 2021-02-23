
###############################Graph######################
##############First Graph######################
dfNodes['max']=dfNodes[['neg','ne','po','m']].max(axis=1)
dfNodes['max_label']=np.argmax(np.array(dfNodes[['neg','ne','po','m']]),axis=1)

dfNodes["sent_rumor"] = dfNodes["rumor_group"]*10 + dfNodes["max_label"]
dfNodes["sent_rumor"] = dfNodes["sent_rumor"].replace([10, 11, 12, 13], 'Rumor')
dfNodes["sent_rumor"] = dfNodes["sent_rumor"].replace([0], 'Negative')
dfNodes["sent_rumor"] = dfNodes["sent_rumor"].replace([1], 'Netural')
dfNodes["sent_rumor"] = dfNodes["sent_rumor"].replace([2], 'Positive')
dfNodes["sent_rumor"] = dfNodes["sent_rumor"].replace([3], 'Mix')

dfNodes["sent_group"]=np.argmax(np.array(dfNodes[['neg','ne','po','m']]),axis=1)
dfNodes["rumor_group"]=np.argmax(np.array(dfNodes[['not_rumor','rumor']]),axis=1)

dfNodes["sent_rumor"]=np.argmax(np.array(dfNodes[['neg','ne','po','m']]),axis=1)
dfNodes["sent_rumor"] = dfNodes.apply(lambda x: x["rumor_group"] if x["rumor_group"]==1 else x["sent_rumor"], axis=1)

dfNodes["sent_rumor_prob"]=dfNodes[['neg','ne','po','m']].max(axis=1)/(dfNodes["neg"]+dfNodes["ne"]+dfNodes["po"]+dfNodes["m"])
dfNodes["sent_rumor_prob"] = dfNodes.apply(lambda x: x["rumorprob"] if x["rumor_group"]==1 else x["sent_rumor_prob"], axis=1)
dfNodes.to_csv('dfNodes.csv')

dfLinks["sent_rumor"] = dfLinks["rumor_group"]*10 + dfNodes["max_label"]
dfLinks["sent_rumor"] = dfLinks["sent_rumor"].replace([10, 11, 12, 13], 'Rumor')
dfLinks["sent_rumor"] = dfLinks["sent_rumor"].replace([0], 'Negative')
dfLinks["sent_rumor"] = dfLinks["sent_rumor"].replace([1], 'Netural')
dfLinks["sent_rumor"] = dfLinks["sent_rumor"].replace([2], 'Positive')
dfLinks["sent_rumor"] = dfLinks["sent_rumor"].replace([3], 'Mix')

dfLinks["sent_group"]=np.argmax(np.array(dfLinks[['neg','ne','po','m']]),axis=1)
dfLinks["rumor_group"]=np.argmax(np.array(dfLinks[['not_rumor','rumor']]),axis=1)

dfLinks["sent_rumor"]=np.argmax(np.array(dfLinks[['neg','ne','po','m']]),axis=1)
dfLinks["sent_rumor"] = dfLinks.apply(lambda x: x["rumor_group"] if x["rumor_group"]==1 else x["sent_rumor"], axis=1)
dfLinks["sent_rumor_prob"]=dfLinks[['neg','ne','po','m']].max(axis=1)/(dfLinks["neg"]+dfLinks["ne"]+dfLinks["po"]+dfLinks["m"])
dfLinks["sent_rumor_prob"] = dfLinks.apply(lambda x: x["rumorprob"] if x["rumor_group"]==1 else x["sent_rumor_prob"], axis=1)

########Code below is using R kernel#########
install.packages('networkD3')
library(networkD3)

dfLinks$max <- paste(dfLinks$max, dfLinks$max_label, sep = add)
dfNodes <-  dfNodes[sample(nrow(dfNodes), 300), ]

color <- c("orange")
linkcolor <- color[(dfLinks$max_label) +1]

dfLinks$sent_rumor[dfLinks$sent_rumor == 10] <- "black"
dfLinks$sent_rumor[dfLinks$sent_rumor == 11] <- "black"
dfLinks$sent_rumor[dfLinks$sent_rumor == 12] <- "black"
dfLinks$sent_rumor[dfLinks$sent_rumor == 13] <- "black"
dfLinks$sent_rumor[dfLinks$sent_rumor == 0] <- "lightblue" #negative
dfLinks$sent_rumor[dfLinks$sent_rumor == 1] <- "lightgreen" #neutral
dfLinks$sent_rumor[dfLinks$sent_rumor == 2] <- "red" #positive
dfLinks$sent_rumor[dfLinks$sent_rumor == 3] <- "orange"

dfNodes$sent_rumor[dfNodes$sent_rumor == 10] <- "black"
dfNodes$sent_rumor[dfNodes$sent_rumor == 11] <- "black"
dfNodes$sent_rumor[dfNodes$sent_rumor == 12] <- "black"
dfNodes$sent_rumor[dfNodes$sent_rumor == 13] <- "black"
dfNodes$sent_rumor[dfNodes$sent_rumor == 0] <- "lightblue"
dfNodes$sent_rumor[dfNodes$sent_rumor == 1] <-  "lightgreen" 
dfNodes$sent_rumor[dfNodes$sent_rumor == 2] <- "red"
dfNodes$sent_rumor[dfNodes$sent_rumor == 3] <- "orange"
dfNodes["sent_rumor"] = dfNodes["sent_rumor"].replace([10, 11, 12, 13], 'Rumor')
dfNodes["sent_rumor"] = dfNodes["sent_rumor"].replace([0], 'Negative')
dfNodes["sent_rumor"] = dfNodes["sent_rumor"].replace([1], 'Netural')
dfNodes["sent_rumor"] = dfNodes["sent_rumor"].replace([2], 'Positive')
dfNodes["sent_rumor"] = dfNodes["sent_rumor"].replace([3], 'Mix')

library(visNetwork)
library(igraph)
library(igraphdata)
library(stringr)
library(rpart)
library(sparkline)


Newnodes <- data.frame(id=dfNodes$tag, 
                       label = dfNodes$tag, 
                       #neutral: weight - 
                       value = dfNodes$sent_rumor_prob*10,
                       group = paste("Group",dfNodes$sent_rumor_prob),
                       
                       title = dfNodes$tag,
                       color = dfNodes$sent_rumor
                       )



Newedages <- data.frame(from = dfLinks$v1,
                        to = dfLinks$v2,
                        
                        value = dfLinks$sent_rumor_prob/100,
                        label = paste("weight",dfLinks$max,sep = "-"),
                        width = dfLinks$sent_rumor_prob/100, 
                        width = 0.1, 
                        color = dfLinks$sent_rumor
                        )


network <- visNetwork(Newnodes, Newedages, height = "1400px", width = "100%",
           main = "Sent_Rumor_Network Graph") %>%
  visOptions(highlightNearest = TRUE)%>%
  visInteraction(navigationButtons = TRUE)%>%
  visOptions(manipulation = TRUE)

visSave(network, file = "sent_rumor_network.html", background = "white")


######################Second Graph############################
# assign rumorprob_ proper weights for forcenetworkD3 graphing later.
dfNodes_highrumor$rumorprob_=10*round(round(100*dfNodes_highrumor$rumorprob)/20)
#summary(dfNodes_highrumor$rumorprob_)
dfLinks_highrumor$linkcolor=round(round(100*dfLinks_highrumor$rumorprob)/70)+1
#summary(dfLinks_highrumor$linkcolor)


library(networkD3)
# Plot
colors=JS('d3.scaleOrdinal().domain(["1", "2", "3"]).range(["#000000", "#111111"])')

p = forceNetwork(Links = dfLinks_highrumor, Nodes = dfNodes_highrumor,
            Source = "key1", Target = "key2",
            Value = "max", NodeID = "tag", opacity = 0.8, fontSize = 50,Group='rumor_group',Nodesize='rumor')

saveNetwork(p, file='BGraph.html', selfcontained = TRUE)

p=forceNetwork(Links = dfLinks_highrumor, Nodes = dfNodes_highrumor,
            Source = "key1", Target = "key2",
            Value = "max", NodeID = "tag", opacity = 0.9, fontSize = 50,Group='rumor_group',Nodesize='rumorprob_', colourScale = JS(ColourScale),linkColour=color)

saveNetwork(p, file='DGraph.html', selfcontained = TRUE)

library(dplyr)
library(htmltools)
color<-dfLinks_highrumor$color

# Df:Edges with rumor prob > 0.06. 409 nodes out of 4000+. 2074 edges
# Node Size: Rumor prob
# Node color: Purple: Negtive;  Orange: Positive
# Edge color: Rumor prob pink:rumor prob>0.5, blue:rumor prob<0.3
# Edge width: # of connections

ColourScale <- 'd3.scaleOrdinal()
            .domain(["lions", "tigers"])
           .range(["#694489","#FF6900",]);'

edgecolor<-colorRampPalette(c('#FFF00','FF0000'),bias=dfLinks_highrumor$linkcolor)
edgecolor<-sapply(dfLinks_highrumor$linkcolor)

browsable(
  tagList(
    tags$head(
      tags$style('
        body{background-color: #DAE3F9 !important}
        .nodetext{fill: #000000}
        .legend text{fill: #FF0000}
      ')
    ),
    forceNetwork(Links = dfLinks_highrumor, Nodes = dfNodes_highrumor,
            Source = "key1", Target = "key2",
            Value = "max", NodeID = "tag", opacity = 0.9, fontSize = 50,Group='rumor_group',Nodesize='rumorprob_', colourScale = JS(ColourScale),linkColour=color)
  )
)
