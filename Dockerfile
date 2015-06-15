FROM julia:0.3

ENV JULIA_VER=0.3
ENV JULIA_PKGDIR=/usr/local/.julia/

RUN julia -e 'Pkg.init()'

COPY REQUIRE $JULIA_PKGDIR/v$JULIA_VER/
RUN julia -e 'Pkg.resolve();'

ADD . /usr/src/app
WORKDIR /usr/src/app

CMD cd src && julia ../test/runtests.jl

