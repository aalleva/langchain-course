from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(temperature=0, model="gpt-4.1-nano")

def main():
    print("Hello from langchain-course!")
    information = """
    Diego Armando Maradona[8]​ (Lanús, Buenos Aires, 30 de octubre de 1960-Dique Luján, Buenos Aires, 25 de noviembre de 2020) fue un futbolista y entrenador argentino.[1]​[9]​ Como jugador, se desempeñó en la posición de mediocampista ofensivo o delantero, siendo considerado por numerosos especialistas,[10]​[11]​ exfutbolistas y personalidades internacionales[12]​ como «uno de los mejores futbolistas en la historia».[13]​

    Maradona ha sido catalogado por diferentes medios, como el «mejor jugador en la historia de la Copa Mundial», de la cual fue designado como el mejor jugador en su edición de 1986.[14]​ En los premios a Jugador del Siglo de la FIFA fue seleccionado como el «mejor futbolista del siglo xx» en la votación popular, obtuvo la tercera posición en la votación de los expertos seleccionados por la FIFA,[n. 2]​ y logró la quinta ubicación en la votación realizada por la IFFHS.[15]​ En la edición de los Premios Globe Soccer 2012 fue distinguido como el mejor «Jugador del Siglo xx».[16]​[17]​ Maradona es considerado una de las figuras más históricas de la República Argentina, y uno de sus mayores representantes en el resto del mundo.[18]​[19]​[20]​[21]​ Asimismo, su persona ha sido motivo de las más variadas referencias en la cultura popular argentina y napolitana.[22]​[23]​

    Criado en el barrio bonaerense de Villa Fiorito, fue fichado para las divisiones juveniles de Argentinos Juniors, donde pasó cinco temporadas obteniendo el récord de ser el máximo goleador del campeonato argentino cinco veces consecutivas. En 1981, fue traspasado a Boca Juniors,[24]​ donde obtuvo el Campeonato Metropolitano, su único título en Argentina. Tras el Mundial de España de 1982, Maradona se convirtió en el primer futbolista en lograr el récord de ser el traspaso más caro del mundo dos veces, al ser transferido al Barcelona por 7,30 millones de euros y luego al Napoli de Italia por 12 millones de euros.[25]​ En España, Diego obtendría tres títulos nacionales antes de acabar en Italia en 1984. Allí, Maradona se convirtió en una de las figuras públicas más importantes de Nápoles,[26]​ al llevar al equipo a lograr el scudetto en dos oportunidades (1987, 1990) y la Copa de la UEFA, el único título internacional de la institución. Luego de siete temporadas como napolitano, en la que acabó como el máximo goleador histórico, Maradona abandonó Italia luego de obtener su primer positivo por dopaje en la temporada 1990-91. En la etapa final de su carrera, jugó en Sevilla y Newell's Old Boys para acabar regresando a Boca Juniors en 1995 y terminar de retirarse en 1997.[27]​

    Con la Selección Argentina, Maradona fue campeón del Mundial Juvenil de 1979, y con los mayores del Mundial de México de 1986 como capitán del equipo, en la que protagonizaría una de las actuaciones individuales más destacadas de la historia del deporte,[28]​[29]​[30]​ al anotar los dos célebres goles que dieron la victoria a su selección en el partido contra Inglaterra en los cuartos de final, el primero de ellos conocido como «la mano de Dios» y el segundo como el «Gol del Siglo», señalado por una votación de la FIFA como el mejor en la historia de los mundiales del siglo XX.[31]​ En Italia 1990, Argentina casi repetiría la misma gesta, pero acabaría como subcampeón. Después de tres años de ausencia por sus problemas de dopaje, Maradona regresó para ayudar en la clasificación de Argentina para el Mundial de Estados Unidos de 1994, torneo en el que Diego volvería a dar positivo en drogas al encontrarse efedrina en sus muestras, siendo expulsado de la competición, lo que contribuyó a la posterior eliminación de Argentina en octavos de final.[32]​ Esta sería su última participación a nivel selecciones como jugador.[33]​
    """
    
    summary_template = """
    Given the information {information} about a person I want to create: 
    1. A short summary
    2. Two interesting facts about them
    """
    summary_prompt = PromptTemplate(template=summary_template, input_variables=["information"])
    summary_chain = summary_prompt | llm
    response = summary_chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
