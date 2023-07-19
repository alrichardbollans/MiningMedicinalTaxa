# docs https://github.com/ropensci/europepmc
# https://docs.ropensci.org/europepmc/articles/introducing-europepmc.html
library(europepmc)

x = europepmc::epmc_search(query = '"malaria" OR "2019nCoV"')
