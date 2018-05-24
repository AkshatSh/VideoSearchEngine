#!/bin/bash

function setRoot
{
	local this
	local lls
	local link
	local bin

	this="${BASH_SOURCE-$0}"
	while [ -h "$this" ]; do
	  lls=`ls -ld "$this"`
	  link=`expr "$lls" : '.*-> \(.*\)$'`
	  if expr "$link" : '.*/.*' > /dev/null; then
	    this="$link"
	  else
	    this=`dirname "$this"`/"$link"
	  fi
	done

	# convert relative path to absolute path
	bin=`dirname "$this"`
	script=`basename "$this"`
	bin=`cd "$bin"; pwd`
	#this="$bin/$script"

	echo "$bin";
}

function isLinux
{
	test "$(uname)" = "Linux"
}

function isMac
{
	test "$(uname)" = "Darwin"
}

function isCygwin
{
	os="$(uname)"
	test "${os:0:6}" = "CYGWIN"
}

function xEnabled
{
	if isCygwin
	then
		return false
	fi

	(xterm -e "") 2>&1 > /dev/null 
}

#--------------------------------------------init--------------------------------------
ROOT=${ROOT="$(setRoot)"}


osName="$(uname)"
if [ "$osName" = Darwin ] || [ "${osName:0:6}" = CYGWIN ] || [ "$osName" = Linux ]
#if [ "$osName" = Darwin ] ||  [ "$osName" = Linux ]
then
	true
else
	echo "Unsupported OS. Currently only Linux, Mac and Windows with \ 
	Cygwin are supported"
	#echo "Unsupported OS. Currently only Linux and Mac OS \ 
	#are supported"
	exit;
fi

CLASSPATH_SEPARATOR=":"

if isLinux || isCygwin
then
	workerHosts=$(cat "$ROOT/conf/workers.conf" | sed 's/[ \t]\+//g' | \
		      sed 's/#.*$//g' | sed '/^$/d' | sed 's/:[0-9]\+$//g' | sort | uniq)
	workers=$(cat "$ROOT/conf/workers.conf" | sed 's/[ \t]\+//g' | \
		      sed 's/#.*$//g' | sed '/^$/d')
	if isCygwin
	then
		CLASSPATH_SEPARATOR=";"
	fi
elif isMac
then
	workerHosts=$(cat "$ROOT/conf/workers.conf" | sed -E 's/[ 	]+//g' | \
		      sed -E 's/#.*$//g' | sed -E '/^$/d' | sed -E 's/:[0-9]+//g' | sort | uniq)
	workers=$(cat "$ROOT/conf/workers.conf" | sed -E 's/[ 	]+//g' | \
		      sed -E 's/#.*$//g' | sed -E '/^$/d')
fi

#--------------------------------------------start workers--------------------------------------
echo "Starting workers"

terminal=xterm
titleOption=-title
if ! [ -z "$(which gnome-terminal)" ]
then
	terminal=gnome-terminal
	titleOption=-t
#elif ! [ -z "$(which konsole)" ]
#then
#	terminal=konsole
fi

for worker in $workers
do 
	if isMac
	then
		host=$(echo $worker | sed -E 's/:[0-9]+$//g') 
		port=$(echo $worker | sed -E 's/^[^:]+://g') 
	else
		host=$(echo $worker | sed 's/:[0-9]\+$//g') 
		port=$(echo $worker | sed 's/^[^:]\+://g') 
	fi
	if isLinux || isCygwin
	then
		if ! xEnabled 
		then
		    (exec ssh $host "cd ${ROOT}; source env/bin/activate; pip install -r requirements.txt; cd VideoSearchEngine; python video_util_worker.py ${host}:$port" 2>&1 | \
		    sed "s/^/$host:$port\\$ /g") & 
		else
		    ${terminal} ${titleOption} "Worker: $host:$port. Do not close this window when worker is running." -e \
		    "bash -c \"ssh $host \\\"cd ${ROOT}; source env/bin/activate; pip install -r requirements.txt; cd VideoSearchEngine; python video_util_worker.py ${host}:$port\\\" | \
		    sed \\\"s/^/$host:$port\\\\$ /g\\\" \" " & 
		fi
	else
		#mac

		osascript -e "tell app \"Terminal\"
			do script \"echo -e \\\"\\\\033]0;Worker: $host:$port. Do not close this window when worker is running.\\\\007\\\"; ssh $host \\\"cd ${ROOT}; source env/bin/activate; pip install -r requirements.txt; cd VideoSearchEngine; python video_util_worker.py ${host}:$port \\\"\"
		end tell"
	fi

done

#outputStyle="X"
#if ! xEnabled
#then
#	outputStyle="T"
#fi

javaOptions=

if isCygwin
then
	javaOptions=-Djline.terminal=jline.UnixTerminal
fi



#--------------------------------------------start server--------------------------------------
echo "Finish starting workers, now starting the server"
cd "$ROOT"
#TODO: See if something needs to be done here