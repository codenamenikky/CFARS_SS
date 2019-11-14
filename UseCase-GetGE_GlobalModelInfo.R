rm(list=ls())
library(ggplot2)
library(reshape)
library(plyr)
library(readxl)
source("PhaseIIFunctions.R")

f = "C:/GitHub/CFARS_SS/PhaseIIData/GE_Final_masked_CFARS_Aggregate_Results_Phase1test_05mps (1).xlsx"
d = get_phase_ii_aggregate_data(f)

d = subset(d,Sensor != 'Triton')
r = melt(d,id.vars = c("metric","Project","Company","Region","Terrain","Height",
                       "Season","Distance","Sensor","RefAnemType","Anem2Type",
                       "RefAnemClass","Anem2Class","IEC","threshold","count",
                       "speed"), measure.vars = c(16:18))

wide = cast(r,Company+Project+Region+Terrain+Height+Season+
              Distance+Sensor+RefAnemType+Anem2Type+RefAnemClass+
              Anem2Class+IEC+speed+threshold+count ~ variable+metric)


ge = get_ge_info(f)
ge = subset(ge,Sensor!='Triton')
levels(ge$metric) = c(levels(ge$metric),"RSD_TI_Corr", "RSD_TI_Raw")
ge$metric[which(ge$metric=="RSD vs. Ref TI regression No Correction")] = "RSD_TI_Raw"
ge$metric[which(ge$metric=="RSD vs. Ref TI regression Corrected")] = "RSD_TI_Corr"
ge_data = subset(ge,variable %in% c("m","c") & metric %in% c("RSD_TI_Corr","RSD_TI_Raw"))

ggplot(subset(ge_data,variable %in% c("m","c")))+
  geom_boxplot(aes(x=variable,group=variable,y=value))+
  geom_point(aes(x=variable,y=value))+
  stat_summary(aes(x=variable,y=value),fun.y='mean',geom='point',color="red",shape=3)+
  facet_grid(variable~metric,scales='free')

cmon = cast(ge_data,
            Project+Company+Region+Terrain+Height+
              Season+Distance+Sensor+RefAnemType+Anem2Type+
              RefAnemClass+Anem2Class+IEC ~ metric+variable)

yes = merge(wide,cmon,by=c("Project","Company","Region","Terrain","Height",
                           "Season","Distance","Sensor","RefAnemType","Anem2Type",
                           "RefAnemClass","Anem2Class","IEC"))



pls = apply_global_model(yes)
pls = apply_simple_model(pls)

ge = ddply(pls,.(speed),get_ge_uncertainties)
unc = ddply(pls,.(speed),get_uncertainties)
spread = rbind(ge,unc)

splot = spread
splot$variable <- factor(splot$variable, 
                         levels = c("mean", "std", "char_ti"))

levels(splot$metric) = c(levels(splot$metric),"RSD-corr. TI","RSD TI","Cup TI",
                         "Global, Filt.","Global, Raw","Simple Corr.")

splot$metric[which(splot$metric=='corrTI_RSD_TI')] = "RSD-corr. TI"
splot$metric[which(splot$metric=='Simple')] = "Simple Corr."
splot$metric[which(splot$metric=='RSD_TI')] = "RSD TI"
splot$metric[which(splot$metric=='Ref_TI')] = "Cup TI"
splot$metric[which(splot$metric=='Global_Filtered')] = "Global, Filt."
splot$metric[which(splot$metric=='Global_Raw')] = "Global, Raw"

var.labs <- c("Char. TI", "Mean","Std Dev.")
names(var.labs) <- c("char_ti","mean","std")

ggplot(subset(splot,speed>=3 & speed<=20 & num>3))+
  geom_line(aes(x=speed,y=RMSE,group=metric,color=metric))+
  geom_point(aes(x=speed,y=RMSE,color=metric))+
  facet_wrap(~variable,labeller = labeller(variable = var.labs))+
  scale_y_continuous(breaks=seq(0,0.15,0.02),labels=seq(0,0.15,0.02))+
  scale_x_continuous(breaks=seq(2,20,2),labels=seq(2,20,2))+
  labs(x='wind speed, m/s', y='TI uncertainty estimate',
       title="Reference Speed vs. TI Standard Error Metrics",
       color="Error Metric")
ggsave(filename='RSD_andModelsUncertainties.png',width=7,height=3)

write.csv(subset(splot,speed>=3 & speed<=20 & num>3),
          file = 'ErrorData_VariousSensorsAndModels.csv',
          row.names = F)

cats = read.csv("Copy of 20190328_IEC_Class_TI_limits_PL.csv")

#levels(spread$metric) = c(levels(spread$metric),
#                          "RSD-corr. TI","RSD TI","Cup TI","Anem #2 TI" ,"Ane2_TI")

#ane = subset(spread, metric=='Ref_TI')
#ane$metric = 'Ane2_TI'
#spread = rbind(spread,ane)

y = melt(pls, id.vars = c("Project","Company","Region","Terrain","Height",
                          "Season","Distance","Sensor","RefAnemType","Anem2Type",
                          "RefAnemClass","Anem2Class","IEC","threshold","count",
                          "speed"), measure.vars = c(35,36,38))


levels(y$variable) = c(levels(y$variable),
                       "Global_Filtered","Global_Raw",
                       "Simple")

y$variable[which(y$variable == "char_ti_global_RSD_TI")] = "Global_Filtered"
y$variable[which(y$variable == "char_ti_globalRaw_RSD_TI")] = "Global_Raw"
y$variable[which(y$variable == "char_ti_simple_RSD_TI")] = "Simple"

y$metric = y$variable
y$variable = "char_ti"
test = merge(y,spread,by=c("speed","metric","variable"))

tplot = test
#tplot$metric <- factor(tplot$metric, 
#                       levels = c("Ref. TI","Anem #2 TI","RSD-corr. TI","RSD TI",
#                                  'corrTI_RSD_TI','RSD_TI','Ref_TI'))

#results = NULL
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
ggsave(filename='PhaseII_CategoryDecisionChart_RSDRaw.png',width=7,height=5)

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
                metric %in% c("Ref. TI","Simple") &
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
  scale_y_discrete(labels=c("Ref. TI","Simple Corr."))
ggsave(filename='PhaseII_CategoryDecisionChart_Simple.png',width=7,height=5)

ggplot(subset(results,gain %in% seq(-2,2,0.5) & 
                metric %in% c("Ref. TI","Global_Raw") &
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
  scale_y_discrete(labels=c("Ref. TI","Global Raw."))
ggsave(filename='PhaseII_CategoryDecisionChart_GlobalRaw.png',width=7,height=5)

ggplot(subset(results,gain %in% seq(-2,2,0.5) & 
                metric %in% c("Ref. TI","Global_Filtered") &
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
  scale_y_discrete(labels=c("Ref. TI","Global Filt."))
ggsave(filename='PhaseII_CategoryDecisionChart_GlobalFiltered.png',width=7,height=5)
