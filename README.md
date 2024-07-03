# Avaliação Sprints 4 e 5 - Programa de Bolsas Compass UOL e AWS - UFES/UFLA abril/2024

![Logo](https://s3.sa-east-1.amazonaws.com/remotar-assets-prod/company-profile-covers/cl7god9gt00lx04wg4p2a93zt.jpg)

## 💻 Sobre o projeto

Este projeto consiste em um modelo de machine learning treinado que seja capaz de classificar quartos de hotel em 3 faixas de preço utilizando o dataset [Hotel Reservaions Dataset](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset), o modelo foi treinado com o algoritmo XGBoost, e implmentado em python utilizando AWS Sagemaker. O acesso é feito a partir de uma API rodando em uma instância EC2.

### Escolha do modelo

Após conhecermos mais a fundo o problema e estudarmos os algoritmos mais utilizados para classificações supervisionadas, chegamos à arvore de evolução de alguns algoritmos utilizados, a conclusão de utilizar o XGBoost surgiu após compararmos seus resultados com outros modelos e chegarmos em uma melhor acurácia. 

### Vantagens do XGBoost

Em relação a outros algoritmos que usam a estratégia de arvores se encontra no ponto onde cada árvore é construída sequencialmente, tentando corrigir os erros das árvores anteriores. Durante esse processo, o algoritmo ajusta os pesos das observações, dando mais peso às instâncias que foram incorretamente previstas.

## 📂 Estrutura do Projeto 

```
/SPRINTS-4-5-PB-AWS-ABRIL
├── src/
│   ├── api/
│   │   ├── routes/
│   │   │   └── predict_router.py
│   │   ├── models/
│   │   │   └── model.py
│   │   ├── schemas/
|   │   |   └── schemas.py
│   |   ├── dockerfile
│   │   └── main.py
|   |
│   └── sagemaker/
│       ├── deploy.py
│       ├── invoke_endpoint.py
│       ├── sagemakerlocal
|       └── train.py
|
├── .gitignore
├── requirements.txt
└── README.md
````

## ☁️ Diagrama de arquitetura AWS

![diagrama](https://github.com/Compass-pb-aws-2024-ABRIL/sprints-4-5-pb-aws-abril/raw/main/assets/sprint4-5.jpg)

## 🖥️ Como Usar 

### Instalação

1. Instale os seguintes serviços:
    - Python
    - Docker

2. Clone o repositório e entre na branch "grupo-5":

    ```bash
    git clone -b grupo-7 https://github.com/Compass-pb-aws-2024-ABRIL/sprints-4-5-pb-aws-abril.git

    cd sprints-4-5-pb-aws-abril

    git checkout grupo-5
    ```
3. Prepare a imagem do docker:

    Encontre o arquivo Dockerfile e construa a imagem:
    
    ```bash
    docker build -t <ImageName> .
    ```
    
     Execute o container:
    
    ```bash
    docker run -p 80:80 <ImageName>
    ```

### Uso da API

- A API estará disponível no IP público da instância EC2 na porta 8000:

  ```
  http://seu-ip-publico:8000
  ```

- Para fazer uma predição, envie uma requisição POST para o endpoint /api/v1/predict com os dados necessários, como o exemplo a seguir:

  ```
  {
  "no_of_adults": 2,
  "no_of_children": 1,
  "no_of_weekend_nights": 1,
  "no_of_week_nights": 3,
  "required_car_parking_space": 1,
  "lead_time": 45,
  "arrival_year": 2023,
  "arrival_month": 6,
  "arrival_date": 15,
  "repeated_guest": 0,
  "no_of_previous_cancellations": 0,
  "no_of_previous_bookings_not_canceled": 1,
  "no_of_special_requests": 2,
  "type_of_meal_plan:"1,
  "room_type_reserved": 1,
  "market_segment_type": 0,
  }
  ```
A resposta é entregue no seguinte formato:
```
{
    "result": 1
}
```


## ✨ Experiência em fazer esse projeto

Executar esta projeto foi uma experiência desafiadora, mas de muito aprendizado, todos adiquiriram a experiência prática com ferramentas que nunca haviam utilizado antes. Acreditamos ter entregue um projeto coerente com nossas habilidades atuais mas que ainda vemos muito espaço para aprimoramento!


### ⚠️ Dificuldades

1. **Integração do RDS ao projeto:** Metade dos integrantes da equipe não conseguiram fazer a integração de sua instância RDS, acreditamos ter sido dificuldades 

2. **Treinamento local:** Conseguir executar nosso algoritmo localmente utilizando o Sagemaker foi um desafio de refatoração e compatibilidade, visto que a execução necessitava de um SO Linux.


## ✍🏻 Autores:

| [<img loading="lazy" src="https://avatars.githubusercontent.com/u/150451502?v=4" width=115><br><sub>Emanuel Oliveira de Assis Domingues</sub>](https://github.com/emanuel-oliveirad) |  [<img loading="lazy" src="https://avatars.githubusercontent.com/u/133805936?v=4" width=115><br><sub>Gabriel Andrade Carvalho</sub>](https://github.com/gabrielcarvandrade) |  [<img loading="lazy" src="https://avatars.githubusercontent.com/u/58453291?v=4" width=115><br><sub>Joao Victor de Morais Reis</sub>](https://github.com/jvmoraisreis) | [<img loading="lazy" src="https://avatars.githubusercontent.com/u/106821885?v=4" width=115><br><sub>Leandro Rodrigues de Paula Castro</sub>](https://github.com/lrpc75)
| :---: | :---: | :---: | :---: |