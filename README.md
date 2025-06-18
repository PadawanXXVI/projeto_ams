# Projeto de Classificação Binária: Predição de Evasão Escolar

Este repositório apresenta um projeto de classificação binária baseado na metodologia **CRISP-DM**, com foco na predição de risco de evasão escolar. O objetivo é desenvolver um modelo supervisionado que identifique estudantes com baixo desempenho (proxy de evasão), utilizando dados educacionais reais.

---

## 📄 Contexto Acadêmico

Este projeto foi desenvolvido para a disciplina de **Aprendizado de Máquina Supervisionado**, ministrada pelo Prof. Dr. **Jairo Simão Santana Melo**, durante o 3º semestre do curso de **Tecnologia em Ciência de Dados** da **Faculdade de Tecnologia e Inovação Senac DF**. Avaliado como parte da **nota final da disciplina**.

**Autores:**

* Anderson de Matos Guimarães
* Renan Ost

---

## 🌐 Dataset: Students' Academic Performance Dataset

* Fonte: Kaggle / Universidade da Jordânia
* LMS usado: Kalboard 360 (xAPI)
* Registros: 480 estudantes
* Variável-alvo original: `Class` (`H`, `M`, `L`)
* Conversão para binário: `L` = 1 (alto risco de evasão), `M`/`H` = 0 (baixo risco)
* Características:

  * Demográficas (gênero, nacionalidade, local de nascimento)
  * Acadêmicas (série, semestre, disciplina)
  * Comportamentais (acessos, mão levantada, visualização de anúncio)
  * Participação dos pais (resposta de questionários, satisfação escolar)
  * Frequência escolar (acima ou abaixo de 7 faltas)

**Referências bibliográficas:**

* Amrieh, E. A., Hamtini, T., & Aljarah, I. (2016). *Mining Educational Data to Predict Student’s academic Performance using Ensemble Methods*. Int. J. of Database Theory and Application, 9(8), 119-136.
* Amrieh, E. A., Hamtini, T., & Aljarah, I. (2015). *Preprocessing and analyzing educational data set using X-API for improving student's performance*. IEEE AEECT.

---

## ⚖️ Metodologia: CRISP-DM

### 1. Entendimento do Negócio

Detectar alunos com alto risco de evasão escolar com base no baixo desempenho (classe `L`).

### 2. Entendimento dos Dados

Análise exploratória com `pandas`, visualizações com `matplotlib`, e verificação de distribuição da variável-alvo.

### 3. Preparação dos Dados

* Remoção de duplicatas
* Conversão da coluna `Class` para binária
* Codificação de variáveis categóricas com `LabelEncoder`

### 4. Modelagem

* Algoritmos usados:

  * RandomForestClassifier
  * LogisticRegression
  * GaussianNB
* Seleção do melhor modelo com base na **acurácia**

### 5. Avaliação

* `accuracy_score` do `scikit-learn`
* Impressão da melhor acurácia e modelo correspondente

### 6. Implantação

* Salvamento do modelo com `pickle`
* Reutilização e validação com novos dados

---

## 🚀 Como executar localmente

1. Clone o repositório:

```bash
git clone https://github.com/PadawanXXVI/projeto_ams.git
cd projeto_ams
```

2. Crie e ative o ambiente virtual:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Execute o script principal:

```bash
python projeto_evasao.py
```

---

## 📁 Estrutura do Repositório

```
projeto_ams/
├── projeto_evasao.py
├── xAPI-Edu-Data.csv
├── requirements.txt
├── .gitignore
└── modelo_evasao.pickle
```

---

## 🎓 Licença e uso acadêmico

Este projeto foi desenvolvido exclusivamente para fins educacionais, com base em dados públicos disponíveis no Kaggle.

---

## 📄 Citação recomendada

Por gentileza, cite os autores do dataset conforme abaixo ao reutilizar:

> Amrieh, E. A., Hamtini, T., & Aljarah, I. (2016). Mining Educational Data to Predict Student’s academic Performance using Ensemble Methods. IJDT, 9(8), 119-136.

> Amrieh, E. A., Hamtini, T., & Aljarah, I. (2015). Preprocessing and analyzing educational data set using X-API. IEEE AEECT.
