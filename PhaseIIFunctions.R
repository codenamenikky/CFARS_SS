library(ggplot2)
library(reshape)
library(plyr)
library(readxl)

get_phase_ii_aggregate_data<-function(filename){
  d = data.frame(read_excel(filename, sheet='TI_values'))
  names(d)[1] = "metric"
  d = d[-1,]
  d = subset(d,Sensor != 'Triton')
  r = melt(d,id.vars = c("metric","Project","Company","Region","Terrain","Height",
                         "Season","Distance","Sensor","RefAnemType","Anem2Type",
                         "RefAnemClass","Anem2Class","IEC"), measure.vars = c(4:80))
  
  r = cbind(r,colsplit(r$variable,split="_",names=c('var','speed','unit')))
  r$variable<-NULL
  r$unit<-NULL
  d = cast(r,metric+Company+Project+Region+Terrain+Height+Season+
             Distance+Sensor+RefAnemType+Anem2Type+RefAnemClass+
             Anem2Class+IEC+speed ~ var)
  d$char_ti = d$mean+1.28*d$std
  d = data.frame(d)
  
  ti_count = data.frame(read_excel(filename, sheet='TI_count'))
  names(ti_count)[1] = "metric"
  ti_count = ti_count[-1,]
  ti_count = subset(ti_count, Sensor != 'Triton')
  
  tic = melt(ti_count,id.vars = c("metric","Project","Company","Region","Terrain","Height",
                                  "Season","Distance","Sensor","RefAnemType","Anem2Type",
                                  "RefAnemClass","Anem2Class","IEC"), measure.vars = c(4:41))
  
  tic = cbind(tic,colsplit(tic$variable,split="_",names=c('var','speed','unit')))
  tic$variable<-NULL
  tic$unit<-NULL
  ti_count = cast(tic,metric+Company+Project+Region+Terrain+Height+Season+
                    Distance+Sensor+RefAnemType+Anem2Type+RefAnemClass+
                    Anem2Class+IEC+speed ~ var)
  
  ti_count$var = NULL
  ti_count$metric = NULL
  names(ti_count)[length(names(ti_count))] <- "count"
  
  ti_threshold = data.frame(read_excel(filename, sheet='TI_count_threshold'))
  names(ti_threshold)[1] = "metric"
  ti_threshold = ti_threshold[-1,]
  ti_threshold = subset(ti_threshold, Sensor != 'Triton')
  
  tict = melt(ti_threshold,id.vars = c("metric","Project","Company","Region","Terrain","Height",
                                       "Season","Distance","Sensor","RefAnemType","Anem2Type",
                                       "RefAnemClass","Anem2Class","IEC"), measure.vars = c(4))
  tict$variable = NULL
  tict$metric = NULL
  names(tict)[length(names(tict))] <- "threshold"
  
  cmon = merge(d,tict,by=c("Project","Company","Region","Terrain","Height",
                           "Season","Distance","Sensor","RefAnemType","Anem2Type",
                           "RefAnemClass","Anem2Class","IEC"))
  
  cmon = merge(cmon,ti_count,by=c("Project","Company","Region","Terrain","Height",
                                  "Season","Distance","Sensor","RefAnemType","Anem2Type",
                                  "RefAnemClass","Anem2Class","IEC","speed"))
  cmon = subset(cmon,count>=threshold)
  return(cmon)
}

get_uncertainties<-function(df){
  return(data.frame(MAE = c(mean(abs(na.omit(df$char_ti_Ref_TI - df$char_ti_Ane2_TI))),
                            mean(abs(na.omit(df$mean_Ref_TI - df$mean_Ane2_TI))),
                            mean(abs(na.omit(df$std_Ref_TI - df$std_Ane2_TI))),
                            mean(abs(na.omit(df$char_ti_Ref_TI - df$char_ti_corrTI_RSD_TI))),
                            mean(abs(na.omit(df$mean_Ref_TI - df$mean_corrTI_RSD_TI))),
                            mean(abs(na.omit(df$std_Ref_TI - df$std_corrTI_RSD_TI))),
                            mean(abs(na.omit(df$char_ti_Ref_TI - df$char_ti_RSD_TI))),
                            mean(abs(na.omit(df$mean_Ref_TI - df$mean_RSD_TI))),
                            mean(abs(na.omit(df$std_Ref_TI - df$std_RSD_TI)))),
                    RMSE = c(sqrt(mean(na.omit(df$char_ti_Ref_TI - df$char_ti_Ane2_TI)^2)),
                             sqrt(mean(na.omit(df$mean_Ref_TI - df$mean_Ane2_TI)^2)),
                             sqrt(mean(na.omit(df$std_Ref_TI - df$std_Ane2_TI)^2)),
                             sqrt(mean(na.omit(df$char_ti_Ref_TI - df$char_ti_corrTI_RSD_TI)^2)),
                             sqrt(mean(na.omit(df$mean_Ref_TI - df$mean_corrTI_RSD_TI)^2)),
                             sqrt(mean(na.omit(df$std_Ref_TI - df$std_corrTI_RSD_TI)^2)),
                             sqrt(mean(na.omit(df$char_ti_Ref_TI - df$char_ti_RSD_TI)^2)),
                             sqrt(mean(na.omit(df$mean_Ref_TI - df$mean_RSD_TI)^2)),
                             sqrt(mean(na.omit(df$std_Ref_TI - df$std_RSD_TI)^2))),
                    MAD = 1.4826*c(
                      median(abs(na.omit(df$char_ti_Ref_TI - df$char_ti_Ane2_TI))),
                      median(abs(na.omit(df$mean_Ref_TI - df$mean_Ane2_TI))),
                      median(abs(na.omit(df$std_Ref_TI - df$std_Ane2_TI))),
                      median(abs(na.omit(df$char_ti_Ref_TI - df$char_ti_corrTI_RSD_TI))),
                      median(abs(na.omit(df$mean_Ref_TI - df$mean_corrTI_RSD_TI))),
                      median(abs(na.omit(df$std_Ref_TI - df$std_corrTI_RSD_TI))),
                      median(abs(na.omit(df$char_ti_Ref_TI - df$char_ti_RSD_TI))),
                      median(abs(na.omit(df$mean_Ref_TI - df$mean_RSD_TI))),
                      median(abs(na.omit(df$std_Ref_TI - df$std_RSD_TI)))),
                    max = c(max(abs(na.omit(df$char_ti_Ref_TI - df$char_ti_Ane2_TI))),
                            max(abs(na.omit(df$mean_Ref_TI - df$mean_Ane2_TI))),
                            max(abs(na.omit(df$std_Ref_TI - df$std_Ane2_TI))),
                            max(abs(na.omit(df$char_ti_Ref_TI - df$char_ti_corrTI_RSD_TI))),
                            max(abs(na.omit(df$mean_Ref_TI - df$mean_corrTI_RSD_TI))),
                            max(abs(na.omit(df$std_Ref_TI - df$std_corrTI_RSD_TI))),
                            max(abs(na.omit(df$char_ti_Ref_TI - df$char_ti_RSD_TI))),
                            max(abs(na.omit(df$mean_Ref_TI - df$mean_RSD_TI))),
                            max(abs(na.omit(df$std_Ref_TI - df$std_RSD_TI)))),
                    min = c(min(abs(na.omit(df$char_ti_Ref_TI - df$char_ti_Ane2_TI))),
                            min(abs(na.omit(df$mean_Ref_TI - df$mean_Ane2_TI))),
                            min(abs(na.omit(df$std_Ref_TI - df$std_Ane2_TI))),
                            min(abs(na.omit(df$char_ti_Ref_TI - df$char_ti_corrTI_RSD_TI))),
                            min(abs(na.omit(df$mean_Ref_TI - df$mean_corrTI_RSD_TI))),
                            min(abs(na.omit(df$std_Ref_TI - df$std_corrTI_RSD_TI))),
                            min(abs(na.omit(df$char_ti_Ref_TI - df$char_ti_RSD_TI))),
                            min(abs(na.omit(df$mean_Ref_TI - df$mean_RSD_TI))),
                            min(abs(na.omit(df$std_Ref_TI - df$std_RSD_TI)))),
                    num = c(rep(length(na.omit(df$char_ti_Ref_TI - df$char_ti_Ane2_TI)),3),
                            rep(length(na.omit(df$char_ti_Ref_TI - df$char_ti_corrTI_RSD_TI)),3),
                            rep(length(na.omit(df$char_ti_Ref_TI - df$char_ti_RSD_TI)),3)),
                    variable = rep(c("char_ti","mean","std"),3),
                    metric=c(rep("Ref_TI",3),rep("corrTI_RSD_TI",3),rep("RSD_TI",3))))
}

category_function<-function(df,ca,percent=0.7,gain=0){
  percents = df$count / sum(df$count)
  
  for(n in 1:nrow(df)){
    this_ti = df$value[n] + gain*df$RMSE[n]
    this_spd = df$speed[n]
    at = subset(ca,ws==this_spd & category=='A')$char_ti
    bt = subset(ca,ws==this_spd & category=='B')$char_ti
    ct = subset(ca,ws==this_spd & category=='C')$char_ti
    
    if(this_ti<ct){
      df$category[n] = "C"
    } else if (this_ti<bt){
      df$category[n] = "B"
    } else if (this_ti<at){
      df$category[n] = "A"
    } else {
      df$category[n] = "A+"
    }
  }
  
  aplus = sum(percents[which(df$category=='A+')])
  ap = sum(percents[which(df$category=='A')])
  bp = sum(percents[which(df$category=='B')])
  cp = sum(percents[which(df$category=='C')])
  
  if(cp>percent){
    category='C'
  } else if ((cp+bp)>percent){
    category='B'
  } else if ((cp+bp+ap)>percent){
    category='A'
  } else {
    category='A+'
  }
  return(data.frame(category,gain,percent))
}

get_ge_info<-function(filename){
  d = data.frame(read_excel(filename, sheet='Regression'))
  names(d)[1] = "metric"
  d = d[-1,]
  r = melt(d,id.vars = c("metric","Project","Company","Region","Terrain","Height",
                         "Season","Distance","Sensor","RefAnemType","Anem2Type",
                         "RefAnemClass","Anem2Class","IEC"), measure.vars = c(2:5))
  #dat = subset(r,metric=="RSD vs. Ref TI regression Corrected")
  return(r)
}


get_ge_uncertainties<-function(df){
  return(data.frame(MAE = c(mean(abs(na.omit(df$char_ti_Ref_TI - df$char_ti_simple_RSD_TI))),
                            mean(abs(na.omit(df$mean_Ref_TI - df$mean_simple_RSD_TI))),
                            mean(abs(na.omit(df$std_Ref_TI - df$std_RSD_TI))),
                            mean(abs(na.omit(df$char_ti_Ref_TI - df$char_ti_global_RSD_TI))),
                            mean(abs(na.omit(df$mean_Ref_TI - df$mean_global_RSD_TI))),
                            mean(abs(na.omit(df$std_Ref_TI - df$std_corrTI_RSD_TI))),
                            mean(abs(na.omit(df$char_ti_Ref_TI - df$char_ti_globalRaw_RSD_TI))),
                            mean(abs(na.omit(df$mean_Ref_TI - df$mean_globalRaw_RSD_TI))),
                            mean(abs(na.omit(df$std_Ref_TI - df$std_RSD_TI)))),
                    RMSE = c(sqrt(mean(na.omit(df$char_ti_Ref_TI - df$char_ti_simple_RSD_TI)^2)),
                             sqrt(mean(na.omit(df$mean_Ref_TI - df$mean_simple_RSD_TI)^2)),
                             sqrt(mean(na.omit(df$std_Ref_TI - df$std_RSD_TI)^2)),
                      sqrt(mean(na.omit(df$char_ti_Ref_TI - df$char_ti_global_RSD_TI)^2)),
                             sqrt(mean(na.omit(df$mean_Ref_TI - df$mean_global_RSD_TI)^2)),
                             sqrt(mean(na.omit(df$std_Ref_TI - df$std_corrTI_RSD_TI)^2)),
                             sqrt(mean(na.omit(df$char_ti_Ref_TI - df$char_ti_globalRaw_RSD_TI)^2)),
                             sqrt(mean(na.omit(df$mean_Ref_TI - df$mean_globalRaw_RSD_TI)^2)),
                             sqrt(mean(na.omit(df$std_Ref_TI - df$std_RSD_TI)^2))),
                    MAD = 1.4826*c(median(abs(na.omit(df$char_ti_Ref_TI - df$char_ti_simple_RSD_TI))),
                                   median(abs(na.omit(df$mean_Ref_TI - df$mean_simple_RSD_TI))),
                                   median(abs(na.omit(df$std_Ref_TI - df$std_RSD_TI))),
                                   median(abs(na.omit(df$char_ti_Ref_TI - df$char_ti_global_RSD_TI))),
                      median(abs(na.omit(df$mean_Ref_TI - df$mean_global_RSD_TI))),
                      median(abs(na.omit(df$std_Ref_TI - df$std_corrTI_RSD_TI))),
                      median(abs(na.omit(df$char_ti_Ref_TI - df$char_ti_globalRaw_RSD_TI))),
                      median(abs(na.omit(df$mean_Ref_TI - df$mean_globalRaw_RSD_TI))),
                      median(abs(na.omit(df$std_Ref_TI - df$std_RSD_TI)))),
                    max = c(max(abs(na.omit(df$char_ti_Ref_TI - df$char_ti_simple_RSD_TI))),
                            max(abs(na.omit(df$mean_Ref_TI - df$mean_simple_RSD_TI))),
                            max(abs(na.omit(df$std_Ref_TI - df$std_RSD_TI))),
                            max(abs(na.omit(df$char_ti_Ref_TI - df$char_ti_global_RSD_TI))),
                            max(abs(na.omit(df$mean_Ref_TI - df$mean_global_RSD_TI))),
                            max(abs(na.omit(df$std_Ref_TI - df$std_corrTI_RSD_TI))),
                            max(abs(na.omit(df$char_ti_Ref_TI - df$char_ti_globalRaw_RSD_TI))),
                            max(abs(na.omit(df$mean_Ref_TI - df$mean_globalRaw_RSD_TI))),
                            max(abs(na.omit(df$std_Ref_TI - df$std_RSD_TI)))),
                    min = c(min(abs(na.omit(df$char_ti_Ref_TI - df$char_ti_simple_RSD_TI))),
                            min(abs(na.omit(df$mean_Ref_TI - df$mean_simple_RSD_TI))),
                            min(abs(na.omit(df$std_Ref_TI - df$std_RSD_TI))),
                            min(abs(na.omit(df$char_ti_Ref_TI - df$char_ti_global_RSD_TI))),
                            min(abs(na.omit(df$mean_Ref_TI - df$mean_global_RSD_TI))),
                            min(abs(na.omit(df$std_Ref_TI - df$std_corrTI_RSD_TI))),
                            min(abs(na.omit(df$char_ti_Ref_TI - df$char_ti_globalRaw_RSD_TI))),
                            min(abs(na.omit(df$mean_Ref_TI - df$mean_globalRaw_RSD_TI))),
                            min(abs(na.omit(df$std_Ref_TI - df$std_RSD_TI)))),
                    num = c(rep(length(na.omit(df$char_ti_Ref_TI - df$char_ti_simple_RSD_TI)),3),
                      rep(length(na.omit(df$char_ti_Ref_TI - df$char_ti_global_RSD_TI)),3),
                            rep(length(na.omit(df$char_ti_Ref_TI - df$char_ti_globalRaw_RSD_TI)),3)),
                    variable = rep(c("char_ti","mean","std"),3),
                    metric=c(rep("Simple",3),rep("Global_Filtered",3),rep("Global_Raw",3))))
}

apply_global_model<-function(df){
  m = mean(unique(na.omit(df$RSD_TI_Corr_m)))
  c = mean(unique(na.omit(df$RSD_TI_Corr_c)))
  mraw = mean(unique(na.omit(df$RSD_TI_Raw_m)))
  craw = mean(unique(na.omit(df$RSD_TI_Raw_c)))
  df$mean_global_RSD_TI = m*((df$mean_corrTI_RSD_TI - df$RSD_TI_Corr_c)/df$RSD_TI_Corr_m)+c
  df$mean_globalRaw_RSD_TI = mraw*df$mean_RSD_TI+craw
  df$char_ti_global_RSD_TI = df$mean_global_RSD_TI + df$std_corrTI_RSD_TI
  df$char_ti_globalRaw_RSD_TI = df$mean_globalRaw_RSD_TI + df$std_RSD_TI
  return(df)
}

apply_simple_model<-function(df){
  df$mean_simple_RSD_TI = df$RSD_TI_Raw_m*((df$mean_corrTI_RSD_TI - df$RSD_TI_Corr_c)/df$RSD_TI_Corr_m)+df$RSD_TI_Raw_c
  df$char_ti_simple_RSD_TI = df$mean_simple_RSD_TI + df$std_RSD_TI
  return(df)
}
