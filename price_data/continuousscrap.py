
import krakenex
import time
import http
zizi=0;

k = krakenex.API()
c=krakenex.Connection('api.kraken.com', 150)
k.set_connection(c)
k.load_key('kraken.key')

while True:
    zizi=zizi+1;
    print(zizi)
    #for i in range (2*24*60): ConnectionResetError
    while True:
        try:
            data=k.query_public('Time')
            print('Time done')
            data2=k.query_public('Ticker',{'pair':'XXBTZEUR,XETCZEUR,XETHZEUR,XLTCZEUR,XREPZEUR,XZECZEUR'})
            print('Prices done')
            break
        except (ValueError, http.client.HTTPException, ConnectionResetError):
            print('Problem!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      
    
    lahour=data['result']['rfc1123'][15:17]
    laminute=data['result']['rfc1123'][18:20]
    ladate=data['result']['rfc1123'][5:7]
    lannee=data['result']['rfc1123'][12:14]
    lemois=data['result']['rfc1123'][8:11]
    lemois=lemois.replace('Jan','1')
    lemois=lemois.replace('Feb','2')
    lemois=lemois.replace('Mar','3')
    lemois=lemois.replace('Apr','4')
    lemois=lemois.replace('May','5')
    lemois=lemois.replace('Jun','6')
    lemois=lemois.replace('Jul','7')
    lemois=lemois.replace('Aug','8')
    lemois=lemois.replace('Sep','9')
    lemois=lemois.replace('Oct','10')
    lemois=lemois.replace('Nov','11')
    lemois=lemois.replace('Dec','12')
    XBTZEUR=data2['result']['XXBTZEUR']['c'][0]
    XETCZEUR=data2['result']['XETCZEUR']['c'][0]
    XETHZEUR=data2['result']['XETHZEUR']['c'][0]
    XLTCZEUR=data2['result']['XLTCZEUR']['c'][0]
    XREPZEUR=data2['result']['XREPZEUR']['c'][0]
    XZECZEUR=data2['result']['XZECZEUR']['c'][0]

    myfile=open('stockprices_'+lemois+'_'+ladate+'_'+lannee+'.txt',"a")
    print(ladate+' '+lemois+' '+lannee+' '+lahour+' '+laminute+' '+XBTZEUR+' '+XETCZEUR+' '+XETHZEUR+' '+XLTCZEUR+' '+XREPZEUR+' '+XZECZEUR,end="\n", file=myfile)

    myfile.close()
    time.sleep(25)
