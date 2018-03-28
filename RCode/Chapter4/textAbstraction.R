generateCorpus = function(df, col , n = 1) {
  dfs = VectorSource(as.character(df[,col]))
  vc = VCorpus(dfs)
  vc = vc %>% tm_map(content_transformer(tolower)) %>% 
    tm_map(removeWords, stopwords("english")) %>%
    tm_map(stemDocument) %>%
    tm_map(removePunctuation) %>%
    tm_map(stripWhitespace)
  return(vc)
}