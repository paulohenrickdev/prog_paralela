#include <omp.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

//cabeçalho das funçoes
void alocar();
void limparMemoria();
void resolverSistema();
void mostrarResultado();
void escreverMatrizA(double **a, int n);
void escreverVetorB(double *b, int n);
void preencherMatrizA(double **a, int n);
void preencherVetorB(double *b, int n);

int n, t, i;
double **a, *b, *resultado, tempoInicial, tempoFinal, tempoGasto, tempoInicial1, tempoFinal1, tempoGasto1, tempoInicial2, tempoFinal2, tempoGasto2;

int main(int argc, char*argv[]){

    printf("\nSolução do sistema linear (Ax=b) com o Método de Gauss.\n\n");

    if(argc > 1){
		n = atoi(argv[1]);
        t = atoi(argv[2]);
        omp_set_num_threads(t);
		printf("Matriz de ordem: %d\n\n", n);
    }

    alocar(n);
    if(a != NULL){
        resolverSistema();
        limparMemoria();
    }else{
        printf("Erro");
    }
    return 0;
}

void alocar(int n){
    // alocando espaço na memória para a Matriz A
    a = malloc(sizeof(double *) * n);
    for (int i=0; i<n; i++) {
        a[i] = malloc(sizeof(double) * n);
    }

    // alocando espaço na memória para o Vetor b
    b = malloc(sizeof(double) * n);

    // alocando espaço de memória para o vetor resultado
    resultado = malloc(sizeof(double) * n);
    
    preencherMatrizA(a, n);
    preencherVetorB(b, n);
    escreverMatrizA(a, n);
    escreverVetorB(b, n);
}

// limpando a memória alocada
void limparMemoria(){
    for (int i=0; i<n; i++) {
        free(a[i]);
    }
    free(a);
    free(b);
    free(resultado);
}

void resolverSistema(){
    tempoInicial = omp_get_wtime();
    #pragma omp parallel default(none) shared(a, b, n, resultado)
    {
    int ID = omp_get_thread_num();
    printf("ID da Thread (%d)\n", ID);

    // eliminação
    #pragma omp colapse(2)
    for (int i=0; i<n-1; i++) {
        for (int j=i+1; j<n; j++) {
            double x;
            x = -a[j][i] / a[i][i];
            for (int k=0; k<n; k++) {
                a[j][k] += a[i][k] * x;
            }
            b[j] += b[i] * x;
        }
    }

    // substituição regressiva 
    #pragma omp colapse(2)
    for (int i=n-1; i>=0; i--) {
        double x = 0;
        for (int j=i+1; j<n; j++) {
            x += a[i][j] * resultado[j];
        }
        x = -x + b[i];
        resultado[i] = x / a[i][i];
    }
    }
    mostrarResultado();

    // capturar o tempo
    tempoFinal = omp_get_wtime();
    tempoGasto = tempoFinal - tempoInicial;
    printf("\nEsse computador calculou todos os valores em %g (s)\n", tempoGasto + tempoGasto1 + tempoGasto2);
}

void mostrarResultado(){
    printf("\nO reultado é: \n");
    printf("x = ");
    
    float tst = resultado[0];
    int x1 = isnan(tst);
    if(x1 != 0) {
        printf("Erro");
        printf("\n\n");
        return;
    }
    
    for (int i=0; i<n; i++) {
        printf("%0.3lf ", resultado[i]);
    }
    printf("\n\n");
}

void escreverMatrizA(double **a, int n){
    printf("Matriz A: \n");
    for(int i=0; i < n; i++){     
       		for(int j=0; j < n; j++){    
        		printf("%f\t", a[i][j]);
			}
			printf("\n\n");
		} 
}

void escreverVetorB(double *b, int n){
    printf("Vetor b: \n");
    for(int i=0; i < n; i++){     
        printf("%f\t", b[i]);
	}
    printf("\n"); 
}

void preencherMatrizA(double **a, int n){
    srand(time(NULL));
    tempoInicial1 = omp_get_wtime();
    #pragma omp parallel default(none) shared(a, n)
    {
    for(int i=0; i < n; i++){  
        #pragma omp single
       	for(int j=0; j < n; j++){    
            a[i][j] = rand() % 50;
		}
	}
    }
    tempoFinal1 = omp_get_wtime();
    tempoGasto1 = tempoFinal1 - tempoInicial1;
}

void preencherVetorB(double *b, int n){
    tempoInicial2 = omp_get_wtime();
    #pragma omp parallel default(none) shared(b, n)
    {
    srand(time(NULL));
    for(int i=0; i < n; i++){        
        b[i] = rand() % 30;
	}
    }
    tempoFinal2 = omp_get_wtime();
    tempoGasto2 = tempoFinal2 - tempoInicial2;
}
