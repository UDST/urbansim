library('mlogit')

data("Fishing", package = "mlogit")
Fish = mlogit.data(Fishing, varying = c(2:9), shape = "wide", choice = "mode")
write.csv(Fish, file='fish.csv')

print('******************')
print('******************')

mnl = mlogit(mode ~ price + catch - 1, data=Fish)
summary(mnl)
print(mnl$coefficients)

print('******************')
print('******************')

mnl = mlogit(mode ~ price:income + catch:income + catch * price - 1, data=Fish)
summary(mnl)
print(mnl$coefficients)

print('******************')
print('******************')

data('TravelMode', package='AER')
TravelMode = mlogit.data(TravelMode, shape='long', choice='choice', varying=c(3:7), alt.var='mode')
write.csv(TravelMode, file='travel_mode.csv')

print('******************')
print('******************')

mnl = mlogit(choice ~ wait + travel + vcost - 1, data=TravelMode)
summary(mnl)
print(mnl$coefficients)

print('******************')
print('******************')

mnl = mlogit(choice ~ wait + travel + income:vcost + income:gcost - 1, data=TravelMode)
summary(mnl)
print(mnl$coefficients)
