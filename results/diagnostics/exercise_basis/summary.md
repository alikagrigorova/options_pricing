# Exercise-rule basis: weight 3 vs weight 9 (60 replications, default MultiConfig otherwise)

Bias = mean − reference (put_fd_greeks); t = bias / (sd/√60).

```
K=36 sigma=0.2 T=1.0 r=0.0 d=0.0
   w3: price +0.0058 t=+19.1 | delta -0.0016 t=-7.0 | gamma +0.0001 t=+0.2 | vega +0.0048 t=+0.1 | rho +0.3093 t=+2.6 | rho_d -0.2560 t=-3.0 | vanna -0.0245 t=-0.8
   w9: price +0.0033 t=+9.9 | delta -0.0006 t=-2.5 | gamma +0.0003 t=+1.1 | vega +0.3295 t=+7.9 | rho +0.1538 t=+1.7 | rho_d -0.1043 t=-1.0 | vanna -0.1114 t=-3.7
K=40 sigma=0.2 T=1.0 r=0.0 d=0.0
   w3: price +0.0081 t=+14.4 | delta -0.0017 t=-5.5 | gamma +0.0007 t=+1.7 | vega -0.0678 t=-1.4 | rho +0.3422 t=+1.9 | rho_d -0.2311 t=-1.2 | vanna -0.0020 t=-0.0
   w9: price +0.0056 t=+12.4 | delta -0.0014 t=-4.2 | gamma +0.0001 t=+0.3 | vega +0.7692 t=+11.0 | rho +0.3441 t=+2.0 | rho_d -0.5442 t=-2.8 | vanna -0.1668 t=-3.5
K=44 sigma=0.2 T=1.0 r=0.0 d=0.0
   w3: price +0.0141 t=+19.2 | delta -0.0023 t=-4.6 | gamma -0.0001 t=-0.1 | vega +0.0079 t=+0.1 | rho +1.4483 t=+5.5 | rho_d -0.8660 t=-3.5 | vanna -0.1221 t=-1.8
   w9: price +0.0096 t=+13.4 | delta -0.0015 t=-3.0 | gamma -0.0008 t=-1.3 | vega +1.3332 t=+10.7 | rho +0.7411 t=+2.8 | rho_d -0.9843 t=-3.8 | vanna -0.3077 t=-4.6
K=36 sigma=0.2 T=1.0 r=0.06 d=0.06
   w3: price +0.0025 t=+6.4 | delta -0.0009 t=-3.2 | gamma +0.0002 t=+0.8 | vega -0.0234 t=-0.4 | rho -0.1646 t=-1.6 | rho_d +0.0900 t=+0.7 | vanna +0.0652 t=+2.0
   w9: price +0.0020 t=+4.6 | delta -0.0008 t=-2.6 | gamma -0.0001 t=-0.3 | vega +0.2084 t=+5.1 | rho -0.0839 t=-0.7 | rho_d +0.0421 t=+0.4 | vanna -0.0591 t=-1.8
K=40 sigma=0.2 T=1.0 r=0.06 d=0.06
   w3: price +0.0047 t=+5.3 | delta -0.0012 t=-2.6 | gamma -0.0009 t=-1.8 | vega +0.1637 t=+1.4 | rho -0.1105 t=-0.5 | rho_d +0.1840 t=+0.8 | vanna +0.0330 t=+0.6
   w9: price +0.0013 t=+1.6 | delta -0.0003 t=-0.5 | gamma -0.0013 t=-2.9 | vega +0.0927 t=+1.0 | rho -0.6585 t=-3.3 | rho_d +0.0374 t=+0.2 | vanna -0.1604 t=-2.6
K=44 sigma=0.2 T=1.0 r=0.06 d=0.06
   w3: price +0.0032 t=+2.9 | delta -0.0015 t=-1.8 | gamma -0.0006 t=-1.1 | vega -0.0745 t=-0.6 | rho +0.2492 t=+0.7 | rho_d +0.2262 t=+0.6 | vanna +0.0427 t=+0.4
   w9: price +0.0009 t=+0.9 | delta +0.0009 t=+1.4 | gamma -0.0008 t=-1.0 | vega -0.1634 t=-1.2 | rho +0.4209 t=+1.3 | rho_d -0.2231 t=-0.6 | vanna -0.1452 t=-1.9
K=36 sigma=0.4 T=1.0 r=0.06 d=0.0
   w3: price +0.0026 t=+2.0 | delta -0.0005 t=-1.3 | gamma +0.0001 t=+0.3 | vega -0.0082 t=-0.1 | rho +0.2162 t=+0.7 | rho_d +0.0726 t=+0.2 | vanna -0.0098 t=-0.4
   w9: price +0.0025 t=+2.0 | delta +0.0003 t=+0.9 | gamma -0.0001 t=-0.8 | vega +0.0520 t=+0.9 | rho +0.0948 t=+0.3 | rho_d +0.1863 t=+0.6 | vanna -0.0318 t=-1.5
K=40 sigma=0.4 T=1.0 r=0.06 d=0.0
   w3: price -0.0008 t=-0.5 | delta -0.0004 t=-0.9 | gamma +0.0001 t=+0.4 | vega -0.0589 t=-0.6 | rho +0.3252 t=+0.6 | rho_d -0.1555 t=-0.4 | vanna -0.0025 t=-0.1
   w9: price +0.0036 t=+2.0 | delta +0.0000 t=+0.0 | gamma -0.0001 t=-0.2 | vega +0.1394 t=+1.8 | rho -0.2972 t=-0.6 | rho_d +0.3743 t=+0.9 | vanna -0.0362 t=-1.4
K=44 sigma=0.2 T=1.0 r=0.06 d=0.0
   w3: price -0.0006 t=-0.4 | delta +0.0008 t=+0.8 | gamma +0.0002 t=+0.2 | vega +0.3751 t=+2.1 | rho +0.0174 t=+0.0 | rho_d -0.6221 t=-1.3 | vanna -0.1403 t=-1.3
   w9: price +0.0051 t=+3.2 | delta +0.0008 t=+0.8 | gamma +0.0010 t=+1.1 | vega +0.5413 t=+3.0 | rho -0.3307 t=-0.7 | rho_d -0.2211 t=-0.5 | vanna +0.1123 t=+1.0
K=36 sigma=0.1 T=0.5 r=0.06 d=0.0
   w3: price +0.0002 t=+5.5 | delta -0.0003 t=-4.6 | gamma +0.0004 t=+2.3 | vega +0.0041 t=+0.4 | rho -0.0062 t=-0.5 | rho_d -0.0056 t=-0.4 | vanna +0.0111 t=+0.6
   w9: price +0.0002 t=+4.1 | delta -0.0002 t=-2.9 | gamma +0.0003 t=+1.6 | vega +0.0333 t=+2.5 | rho -0.0061 t=-0.4 | rho_d +0.0056 t=+0.3 | vanna -0.0564 t=-2.8
K=40 sigma=0.2 T=1.0 r=0.06 d=0.0
   w3: price -0.0003 t=-0.3 | delta +0.0000 t=+0.0 | gamma +0.0002 t=+0.3 | vega +0.0015 t=+0.0 | rho +0.3002 t=+1.0 | rho_d +0.1078 t=+0.4 | vanna +0.0667 t=+1.0
   w9: price +0.0040 t=+4.2 | delta -0.0018 t=-3.3 | gamma -0.0009 t=-1.5 | vega +0.2521 t=+2.4 | rho -0.1835 t=-0.6 | rho_d +0.0822 t=+0.3 | vanna -0.0832 t=-1.3

```
