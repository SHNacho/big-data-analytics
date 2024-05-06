FROM apache/spark:3.5.1-scala2.12-java17-python3-r-ubuntu

USER root

RUN echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | tee /etc/apt/sources.list.d/sbt.list \
&& echo "deb https://repo.scala-sbt.org/scalasbt/debian /" | tee /etc/apt/sources.list.d/sbt_old.list \
&& curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | apt-key add \
&& apt-get update \
&& apt-get install -y sbt openjdk-17-source

RUN ln -s /bin/python3 /bin/python