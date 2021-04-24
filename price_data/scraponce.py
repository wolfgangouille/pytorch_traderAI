#!/usr/bin/env python2

# This file is part of krakenex.
# Licensed under the Simplified BSD license. See examples/LICENSE.

# Demonstrates how to use the conditional close functionality; that is,
# placing an order that, once filled, will place another order.
#
# This can be useful for very simple automation, where a bot is not
# needed to constantly monitor execution.
import krakenex
import time
k = krakenex.API()
k.load_key('kraken.key')

data=k.query_public('Time')
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


data=k.query_public('Ticker',{'pair':'XXBTZEUR,XETCZEUR,XETHZEUR,XLTCZEUR,XREPZEUR,XZECZEUR'})
XBTZEUR=data['result']['XXBTZEUR']['c'][0]
XETCZEUR=data['result']['XETCZEUR']['c'][0]
XETHZEUR=data['result']['XETHZEUR']['c'][0]
XLTCZEUR=data['result']['XLTCZEUR']['c'][0]
XREPZEUR=data['result']['XREPZEUR']['c'][0]
XZECZEUR=data['result']['XZECZEUR']['c'][0]

myfile=open("scarplog.txt","a")

print(ladate+' '+lemois+' '+lannee+' '+lahour+' '+laminute+' '+XBTZEUR+' '+XETCZEUR+' '+XETHZEUR+' '+XLTCZEUR+' '+XREPZEUR+' '+XZECZEUR,end="\n", file=myfile)

myfile.close()
