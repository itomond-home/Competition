"""Geographically Weighted Regression

    緯度経度を設定することで,
    位置情報を付与した回帰分析を行える.

"""

library(spgwr)
library(mlbench)
data(BostonHousing2)

# データの用意
Y <- BostonHousing2$cmedv   # 被説明変数
X <- subset(BostonHousing2, T, c(crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat))
X$chas <- as.numeric(X$chas) - 1
X <- as.matrix(X)   # 説明変数
Sp <- as.matrix( subset(BostonHousing2, T, c(lon, lat)) )      # 緯度・経度

# CVによる最適なバンド幅の選択
h <- gwr.sel(Y~X, coords=Sp)

# 最適なバンド幅を用いたGWR
fit <- gwr(Y~X, coords=Sp, bandwidth=h)

# 結果の図示
est <- fit$SDF$Xage
value <- (est - min(est)) / diff(range(est))
cs <- colorRamp( c("blue", "green", "yellow", "red"), space="rgb")
cols <- rgb( cs(value), maxColorValue=256 )
plot(Sp, col=cols, xlab="Latitude", ylab="Longitude", pch=20, cex=1)
