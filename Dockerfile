FROM continuumio/miniconda3:latest

WORKDIR /workspace

# 1. 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. conda-forge를 통해 MPI, PETSc, OpenMDAO 및 기본 패키지 설치
# conda를 사용하면 까다로운 petsc4py의 의존성(C빌드 등)을 완벽하게 우회하여 설치할 수 있습니다.
RUN conda install -y -c conda-forge \
    python=3.9 \
    mpi4py \
    petsc4py \
    openmdao \
    jupyterlab \
    ipywidgets \
    plotly \
    salib \
    pandas \
    numpy \
    scipy \
    && conda clean -ya

# 3. FAST-UAV 소스 코드 복사 및 의존성 설치
# (개발 모드 -e 옵션으로 설치하여 로컬 코드 수정사항이 바로 반영되도록 설정)
COPY . /workspace/
# fastoad 및 FAST-UAV 프로젝트 설치
RUN pip install -e .

EXPOSE 8888

# 4. Jupyter Lab 실행 (기본 포트 8888, 토큰: fastuav)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token='fastuav'"]
