rm(list=ls())
library(ggplot2)
library(reshape)
library(plyr)
library(readxl)
source("PhaseIIFunctions.R")

f = "C:/GitHub/CFARS_SS/PhaseIIData/GE_Final_masked_CFARS_Aggregate_Results_Phase1test_05mps (1).xlsx"
d = get_phase_ii_aggregate_data(f)

r = melt(d,id.vars = c("metric","Project","Company","Region","Terrain","Height",
                       "Season","Distance","Sensor","RefAnemType","Anem2Type",
                       "RefAnemClass","Anem2Class","IEC","threshold","count",
                       "speed"), measure.vars = c(16:18))

wide = cast(r,Company+Project+Region+Terrain+Height+Season+
              Distance+Sensor+RefAnemType+Anem2Type+RefAnemClass+
              Anem2Class+IEC+speed+threshold+count ~ variable+metric)


ggplot(d)+
  geom_line(aes(x=speed,y=mean,group=metric,color=metric))+
  facet_wrap(~Company+Project,labeller = label_wrap_gen(multi_line = F),ncol=3)+
  #theme(axis.text.x = element_text(size=10),
  #      axis.text.y = element_text(size=10),
  #      strip.text.x = element_text(size=6))+
  labs(y="Characteristic TI",color='Sensor')
ggsave(filename='AllSites_Anem2VsRef_CharacteristicTIcurves.png',width=10,height=5)

ggplot(subset(wide,count>threshold & speed>=3))+
  geom_step(aes(x=char_ti_Ref_TI-char_ti_RSD_TI,y=..y..,color='RSD'),stat="ecdf")+
  geom_step(aes(x=char_ti_Ref_TI-char_ti_corrTI_RSD_TI,y=..y..,color='RSD-corr'),stat="ecdf")+
  geom_step(aes(x=char_ti_Ref_TI-char_ti_Ane2_TI,y=..y..,color='Anems'),stat="ecdf")+
  theme(axis.text.x = element_text(size=10),
        axis.text.y = element_text(size=10))+
  #scale_x_continuous(breaks=round(seq(-0.1,0.1,0.005),3),labels=round(seq(-0.1,0.1,0.005),3))+
  scale_y_continuous(breaks=seq(0,1,0.1),labels=paste0(seq(0,100,10),"%"))+
  geom_abline(aes(slope=0,intercept=0.16),linetype='dashed')+
  geom_abline(aes(slope=0,intercept=0.84),linetype='dashed')+
  geom_text(aes(x=-0.015,y=0.19,label='16%'))+
  geom_text(aes(x=-0.015,y=0.87,label='84%'))+
  labs(x='Anem #1 - Anem #2',
       title='Cumulative Distribution of Anem #1 - Anem #2 Char. TI Values\nWind Speed >= 3 m/s and Bin Count > Bin Threshold')
ggsave(filename='CumDist.png',width=6,height=5)


ggplot(subset(wide,count>threshold & speed>=3))+
  geom_step(aes(x=abs(char_ti_Ref_TI-char_ti_RSD_TI),y=..y..,color='RSD'),stat="ecdf")+
  geom_step(aes(x=abs(char_ti_Ref_TI-char_ti_corrTI_RSD_TI),y=..y..,color='RSD-corr'),stat="ecdf")+
  geom_step(aes(x=abs(char_ti_Ref_TI-char_ti_Ane2_TI),y=..y..,color='Anems'),stat="ecdf")+
  theme(axis.text.x = element_text(size=10),
        axis.text.y = element_text(size=10))+
  #scale_x_continuous(breaks=seq(0,0.3,0.005),labels=seq(0,0.3,0.005))+
  scale_y_continuous(breaks=seq(0,1,0.1),labels=paste0(seq(0,100,10),"%"))+
  geom_abline(aes(slope=0,intercept=0.68),linetype='dashed')+
  geom_abline(aes(slope=0,intercept=0.95),linetype='dashed')+
  geom_text(aes(x=0.0025,y=0.71,label='68%'))+
  geom_text(aes(x=0.0025,y=0.98,label='95%'))+
  labs(x='|Anem #1 - Anem #2|',
       title='Cumulative Distribution of |Anem #1 - Anem #2 Char. TI| Values\nWind Speed >= 3 m/s and Bin Count > Bin Threshold')
ggsave(filename='CumDistAbs.png',width=6,height=5)

spread = ddply(wide,.(speed),get_uncertainties)

ggplot(subset(spread,speed>=3 & speed<=20 & num>=1))+
  geom_line(aes(x=speed,y=RMSE,color='RMSE'))+
  geom_line(aes(x=speed,y=max,color='max'))+
  geom_point(aes(x=speed,y=RMSE,color='RMSE'))+
  geom_point(aes(x=speed,y=max,color='max'))+
  facet_grid(metric~variable)+
  scale_y_continuous(breaks=seq(0,0.15,0.02),labels=seq(0,0.15,0.02))+
  scale_x_continuous(breaks=seq(2,20,2),labels=seq(2,20,2))+
  labs(x='wind speed, m/s', y='TI uncertainty estimate',
       title="Reference Speed vs. TI Standard Error Metrics",
       color="Error Metric")
ggsave(filename='BatchErrorEstimates.png',width=7,height=3)

splot = spread
splot$variable <- factor(splot$variable, 
                          levels = c("mean", "std", "char_ti"))

levels(splot$metric) = c(levels(splot$metric),"RSD-corr. TI","RSD TI","Cup TI","Anem #2 TI")


splot$metric[which(splot$metric=='corrTI_RSD_TI')] = "RSD-corr. TI"
splot$metric[which(splot$metric=='RSD_TI')] = "RSD TI"
splot$metric[which(splot$metric=='Ref_TI')] = "Cup TI"

var.labs <- c("Char. TI", "Mean","Std Dev.")
names(var.labs) <- c("char_ti","mean","std")

ggplot(subset(splot,speed>=3 & speed<=20 & num>1))+
  geom_line(aes(x=speed,y=RMSE,group=metric,color=metric))+
  geom_point(aes(x=speed,y=RMSE,color=metric))+
  facet_wrap(~variable,labeller = labeller(variable = var.labs))+
  scale_y_continuous(breaks=seq(0,0.15,0.02),labels=seq(0,0.15,0.02))+
  scale_x_continuous(breaks=seq(2,20,2),labels=seq(2,20,2))+
  labs(x='wind speed, m/s', y='TI uncertainty estimate',
       title="Reference Speed vs. TI Standard Error Metrics",
       color="Error Metric")

ggsave(filename='ErrorEstimates.png',width=7,height=3)

cats = read.csv("Copy of 20190328_IEC_Class_TI_limits_PL.csv")
ggplot(NULL)+
  geom_line(data=cats,aes(x=ws,y=char_ti,group=category,color=category))+
  geom_line(data=subset(wide, speed>=3),
            aes(x=speed,y=mean_corrTI_RSD_TI,color="RSD"))+
  geom_line(data=subset(wide, speed>=3),
            aes(x=speed,y=mean_RSD_TI,color="RSD"))+
  geom_line(data=subset(wide, speed>=3),
            aes(x=speed,y=mean_Ref_TI,color="Ref"))+
  facet_wrap(~Company+Project)+
  xlim(3,16)
ggsave("TISanityCheck.png",width=7,height=5)



levels(spread$metric) = c(levels(spread$metric),
                          "RSD-corr. TI","RSD TI","Cup TI","Anem #2 TI" ,"Ane2_TI")

ane = subset(spread, metric=='Ref_TI')
ane$metric = 'Ane2_TI'
spread = rbind(spread,ane)

test = merge(r,spread,by=c("speed","metric","variable"))

ggplot(NULL)+
  geom_line(data=subset(cats,ws>=3 & ws<=20),aes(x=ws,y=char_ti,group=category,color=category))+
  geom_line(data=subset(test,variable=='char_ti' & 
                          speed>=3 & speed<=20),
            aes(x=speed,y=value))+
  geom_errorbar(data=subset(test,variable=='char_ti' & 
                              speed>=3 & speed<=20),
                aes(x=speed,ymin=value-RMSE,ymax=value+RMSE),size=0.5)+
  facet_grid(metric~Company+Project,labeller = label_wrap_gen(multi_line = F))+
  labs(x='wind speed bin, m/s',y='Characteristic TI (mean+1.28*std)',
       title='Characteristic TI ("Ref"), with RMSE Errorbars and Category Boundaries')
ggsave(filename="CharacteristicTI_AllSites_RMSEErrorbars.png",width=11,height=8)

ggplot(NULL)+
  geom_line(data=subset(cats,ws>=3 & ws<=20),aes(x=ws,y=char_ti,group=category,color=category))+
  geom_line(data=subset(test,variable=='char_ti' & 
                          speed>=3 & speed<=20),
            aes(x=speed,y=value,group=metric,linetype=metric))+
  geom_errorbar(data=subset(test,variable=='char_ti' & metric=='Ref_TI' &
                              speed>=3 & speed<=20),
                aes(x=speed,ymin=value-RMSE,ymax=value+RMSE),size=0.5)+
  facet_wrap(~Company+Project,labeller = label_wrap_gen(multi_line = F))+
  labs(x='wind speed bin, m/s',y='Characteristic TI (mean+1.28*std)',
       title='Characteristic TI ("Ref"), with RMSE Errorbars and Category Boundaries')


ggplot(NULL)+
   geom_area(data=subset(cats,ws>=3 & ws<=15 & category=='A'),
             aes(x=ws,y=char_ti,fill='A'),alpha=0.2)+
   geom_area(data=subset(cats,ws>=3 & ws<=15 & category=='B'),
             aes(x=ws,y=char_ti,fill='B'),alpha=0.2)+
   geom_area(data=subset(cats,ws>=3 & ws<=15 & category=='C'),
            aes(x=ws,y=char_ti,fill='C'),alpha=0.2)+
  geom_line(data=subset(test,Company=="ULGermany" & Project=="Project2a" & 
                          speed>=3 & speed<=15 & 
                          variable=='char_ti' & metric=='corrTI_RSD_TI'),
            aes(x=speed,y=value))+
  geom_errorbar(data=subset(test,Company=="ULGermany" & Project=="Project2a" & 
                              speed>=3 & speed<=15 & 
                              variable=='char_ti' & metric=='corrTI_RSD_TI'),
                aes(x=speed,ymin=value-RMSE,ymax=value+RMSE),size=0.5)+
  scale_y_continuous(breaks=seq(0.1,0.4,0.05),labels=seq(0.1,0.4,0.05))+
  scale_x_continuous(breaks=1:25,labels=1:25)+
  theme(axis.text.x=element_text(size=13),axis.text.y=element_text(size=13))+
  coord_cartesian(ylim=c(0,0.4),xlim=c(2,15))+
  #facet_wrap(~metric,nrow=1)+
  labs(x='wind speed bin, m/s',y='Characteristic TI',fill='Turbine Class',
       title='Characteristic TI, with RMSE Errorbar &\nTurbine Class Boundaries Boundaries')
ggsave(filename="CharacteristicTI_Example01_RMSEErrorbars_RSDcorr.png",width=7,height=5)


tplot = test
levels(tplot$metric) = c(levels(tplot$metric),"RSD-corr. TI","RSD TI","Ref. TI","Anem #2 TI")
tplot$metric[which(tplot$metric=='corrTI_RSD_TI')] = "RSD-corr. TI"
tplot$metric[which(tplot$metric=='RSD_TI')] = "RSD TI"
tplot$metric[which(tplot$metric=='Ref_TI')] = "Ref. TI"
tplot$metric[which(tplot$metric=='Ane2_TI')] = "Anem #2 TI"

tplot$metric <- factor(tplot$metric, 
                         levels = c("Ref. TI","Anem #2 TI","RSD-corr. TI","RSD TI",
                                    'corrTI_RSD_TI','RSD_TI','Ref_TI'))


ggplot(NULL)+
  geom_area(data=subset(cats,ws>=3 & ws<=15 & category=='A'),
            aes(x=ws,y=char_ti,fill='A'),alpha=0.2)+
  geom_area(data=subset(cats,ws>=3 & ws<=15 & category=='B'),
            aes(x=ws,y=char_ti,fill='B'),alpha=0.2)+
  geom_area(data=subset(cats,ws>=3 & ws<=15 & category=='C'),
            aes(x=ws,y=char_ti,fill='C'),alpha=0.2)+
  geom_line(data=subset(tplot,Company=="ULGermany" & Project=="Project2a" & 
                          speed>=3 & speed<=15 & 
                          variable=='char_ti'),
            aes(x=speed,y=value))+
  geom_errorbar(data=subset(tplot,Company=="ULGermany" & Project=="Project2a" & 
                              speed>=3 & speed<=15 & 
                              variable=='char_ti'),
                aes(x=speed,ymin=value-RMSE,ymax=value+RMSE),size=0.5)+
  scale_y_continuous(breaks=seq(0.1,0.4,0.05),labels=seq(0.1,0.4,0.05))+
  scale_x_continuous(breaks=1:25,labels=1:25)+
  theme(axis.text.x=element_text(size=13),axis.text.y=element_text(size=13))+
  coord_cartesian(ylim=c(0,0.4),xlim=c(2,15))+
  facet_wrap(~metric,nrow=1)+
  labs(x='wind speed bin, m/s',y='Characteristic TI',fill='Turbine Class',
       title='Characteristic TI, with RMSE Errorbar &\nTurbine Class Boundaries Boundaries')
ggsave(filename="CharacteristicTI_Example01_RMSEErrorbars_All.png",width=12,height=5)

results = NULL
for(g in seq(-3,3,0.1)){
  cmon = ddply(subset(tplot,speed %in% seq(5,9,0.5) & variable == "char_ti"),
               .(Company,Project,metric),category_function,cats,
               percent=0.7,gain=g)
  results=rbind(results,cmon)
}

results$category <- factor(results$category, 
                           levels = rev(c("C","B","A","A+")))

myColors <- rev(c("limegreen","deepskyblue2","sienna2","red3"))
names(myColors) <- levels(results$category)
colScale <- scale_fill_manual(name = "category",values = myColors)

ggplot(subset(results,gain %in% seq(-2,2,0.5) & 
                metric %in% c("Ref. TI","RSD TI") &
                paste0(Company,Project) !='GEProject04'))+
  geom_tile(aes(x=gain,y=paste(Company,Project,metric),fill=category),color='black',size=1)+
  scale_x_continuous(breaks=seq(-2,2,0.5),labels=seq(-2,2,0.5))+
  geom_vline(aes(xintercept=-1.25),linetype="dashed",color='white',size=1)+
  geom_vline(aes(xintercept=1.25),linetype="dashed",color='white',size=1)+
  labs(y='Project',x="Standard Errors Perturbation")+
  colScale+
  #theme(axis.text.y = element_text(size=6),
  #      axis.text.x = element_text(size=13),
  #      legend.text = element_text(size=13))+
  facet_wrap(~Company+Project,ncol=1,scale='free_y')+
  theme(strip.background = element_blank(),strip.text.x = element_blank())+
  scale_y_discrete(labels=c("Ref. TI","RSD TI"))
ggsave(filename='PhaseII_CategoryDecisionChart_RSD.png',width=7,height=5)

ggplot(subset(results,gain %in% seq(-2,2,0.5) & 
                metric %in% c("Ref. TI","RSD-corr. TI") &
                paste0(Company,Project) !='GEProject04'))+
  geom_tile(aes(x=gain,y=paste(Company,Project,metric),fill=category),color='black',size=1)+
  scale_x_continuous(breaks=seq(-2,2,0.5),labels=seq(-2,2,0.5))+
  geom_vline(aes(xintercept=-1.25),linetype="dashed",color='white',size=1)+
  geom_vline(aes(xintercept=1.25),linetype="dashed",color='white',size=1)+
  labs(y='Project',x="Standard Errors Perturbation")+
  colScale+
  #theme(axis.text.y = element_text(size=6),
  #      axis.text.x = element_text(size=13),
  #      legend.text = element_text(size=13))+
  facet_wrap(~Company+Project,ncol=1,scale='free_y')+
  theme(strip.background = element_blank(),strip.text.x = element_blank())+
  scale_y_discrete(labels=c("Ref. TI","RSD-corr. TI"))
ggsave(filename='PhaseII_CategoryDecisionChart_RSDCorr.png',width=7,height=5)

ggplot(subset(results,gain %in% seq(-2,2,0.5) & 
                metric %in% c("Ref. TI","Anem #2 TI") &
                paste0(Company,Project) !='GEProject04'))+
  geom_tile(aes(x=gain,y=paste(Company,Project,metric),fill=category),color='black',size=1)+
  scale_x_continuous(breaks=seq(-2,2,0.5),labels=seq(-2,2,0.5))+
  geom_vline(aes(xintercept=-1.25),linetype="dashed",color='white',size=1)+
  geom_vline(aes(xintercept=1.25),linetype="dashed",color='white',size=1)+
  labs(y='Project',x="Standard Errors Perturbation")+
  colScale+
  #theme(axis.text.y = element_text(size=6),
  #      axis.text.x = element_text(size=13),
  #      legend.text = element_text(size=13))+
  facet_wrap(~Company+Project,ncol=1,scale='free_y')+
  theme(strip.background = element_blank(),strip.text.x = element_blank())+
  scale_y_discrete(labels=c("Ref. TI","Anem #2 TI"))
ggsave(filename='PhaseII_CategoryDecisionChart_Anem2.png',width=7,height=5)


ggplot(subset(results,gain %in% seq(-2,2,0.5) & 
                metric %in% c("Ref_TI","corrTI_RSD_TI")))+
  geom_tile(aes(x=gain,y=paste(Company,Project,metric),fill=category),color='black',size=1)+
  scale_x_continuous(breaks=seq(-2,2,0.5),labels=seq(-2,2,0.5))+
  geom_vline(aes(xintercept=-1.25),linetype="dashed",color='white',size=1)+
  geom_vline(aes(xintercept=1.25),linetype="dashed",color='white',size=1)+
  labs(y='Project',x="Standard Errors Perturbation")+
  colScale+
  #theme(axis.text.y = element_text(size=6),
  #      axis.text.x = element_text(size=13),
  #      legend.text = element_text(size=13))+
  facet_wrap(~Company+Project,ncol=1,scale='free_y')+
  theme(strip.background = element_blank(),strip.text.x = element_blank())
ggsave(filename='CategoryDecisionChart.png',width=7,height=5)


my_fun<-function(df,this_gain){
  df = subset(df,abs(gain)<this_gain)
  if(length(unique(df$category))>1){
    return(data.frame(gain=this_gain,change=1))
  } else {
    return(data.frame(gain=this_gain,change=0))
  }
}

dat = NULL
for(gain in seq(0,3,0.1)){
  yes = ddply(results,.(Company,Project,metric),my_fun,gain)
  dat = rbind(dat,yes)
}

nrow(yes)
my_fun=function(val) return(sum(val)/12)

ggplot(dat)+
  stat_summary(aes(x=gain,y=change,group=metric,color=metric),fun.y=my_fun,geom='point')+
  stat_summary(aes(x=gain,y=change,group=metric,color=metric),fun.y=my_fun,geom='line')+
  #scale_y_continuous(breaks=seq(0,0.4,0.05),labels=paste0(seq(0,40,5),"%"))+
  #scale_x_continuous(limits=c(0,2),breaks=seq(0,2,0.5),labels=seq(0,2,0.5))+
  geom_vline(aes(xintercept=1),linetype='dashed')+
  theme(axis.text.x=element_text(size=12),axis.text.y=element_text(size=12))+
  labs(x="Standard Errors (RMSE)",
       y="Percent Sites with Category Decision Changes",
       title='Anemometer Standard Error vs. Turbine Class Decision Error')
ggsave(filename='AnemErrorVsClassDecisionError.png',width=6,height=5)



ggplot(dat)+
  stat_summary(aes(x=gain,y=change),fun.y=my_fun,geom='point')+
  stat_summary(aes(x=gain,y=change),fun.y=my_fun,geom='line')+
  scale_y_continuous(breaks=seq(0,0.4,0.05),labels=paste0(seq(0,40,5),"%"))+
  scale_x_continuous(breaks=seq(0,4,0.5),labels=seq(0,4,0.5))+
  geom_vline(aes(xintercept=1),linetype='dashed')+
  theme(axis.text.x=element_text(size=12),axis.text.y=element_text(size=12))+
  labs(x="Standard Errors (RMSE)",
       y="Percent Sites with Category Decision Changes",
       title='Anemometer Standard Error vs. Turbine Class Decision Error')


head(dr)
my_fun<-function(dat){
  return(data.frame(percent=sum(subset(dat,speed>=7 & speed<=12)$count)/sum(dat$count)))
}
mean(ddply(dr,.(Company,Project),my_fun)$percent)

subset(dr,Company=='UL' & Project=='04a')[,c("speed","count")]
