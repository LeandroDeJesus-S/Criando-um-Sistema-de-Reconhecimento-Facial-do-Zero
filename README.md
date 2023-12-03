# Sistema de Reconhecimento Facial com Tensorflow
### Desafio de projeto DIO "Criando um Sistema de Reconhecimento Facial do Zero"

### Funcionamento básico
- Para detecção das faces foi utilizada uma rede MTCNN e para classificação foi realizado a transferência de aprendizado da rede MobileNetV2

### requirements
- python >= 3.10 

### usage
1. Crie um ambiente virtual
   ```bash
   python -m venv venv  # powershell
   python3 -m venv venv  # linux (sudo apt install python3-pip caso não tenha)
   ```
2. Ative o ambiente criado
   ```bash
   venv/scripts/activate # powershell
   . venv/bin/activate  # linux
   ```

3. Instale as dependências
   ```bash
   pip install -r requirements.txt  # use pip3 se estiver no linux
   ```

4. Coloque suas imagens de cada classe na pasta 'train' e 'test' no formato:
   - train/
      - class_0/
        - img.jpg
      - class_1/
        - img.jpg
      - ...

5. Configure as constantes do arquivo program.py conforme necessário
6. Ao finalizar o treinamento rode o programa.
   ```bash
   python program.py  # python3 ao invés de python para linux
   ```        
  