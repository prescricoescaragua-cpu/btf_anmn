/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */
import { GoogleGenAI } from "@google/genai";

// --- STATE ---
let activeTemplateId: string = 'adult-routine';
let abortController: AbortController | null = null;
let isGenerating = false;

// --- CONSTANTS ---
const API_KEY = process.env.API_KEY as string;
const ai = new GoogleGenAI({ apiKey: API_KEY });

const BASE_INSTRUCTION = `
Você é um assistente médico especialista em semiologia. Sua tarefa é organizar as seguintes anotações de um profissional de saúde em um registro de anamnese estruturado, seguindo estritamente o modelo e as instruções fornecidas abaixo.
Você deve interpretar as anotações, reescrevê-las e sintetizá-las em um texto coeso e profissional, utilizando terminologia médica adequada.

---
REGRAS GERAIS
---
1.  **Siga o Modelo:** Preencha as seções APENAS com as informações presentes nas anotações. Siga as diretrizes de cada campo do modelo sobre o que fazer quando a informação não estiver disponível (ex: omitir o campo ou a seção).
2.  **Formatação:** A resposta deve ser clara, concisa e formatada profissionalmente, utilizando a estrutura exata do modelo. Use Markdown para formatação (ex: ## Título, - item de lista).
3.  **Expansão de Termos:** Você DEVE expandir abreviações de exame físico para uma descrição semiológica completa e padronizada. NÃO escreva apenas "exame normal". Transforme a abreviação na descrição completa.
    - **Exemplos Obrigatórios:**
    - "ar: normal" DEVE ser expandido para "AR: Murmúrio vesicular universalmente audível, sem ruídos adventícios."
    - "acv: normal" DEVE ser expandido para "ACV: Ritmo cardíaco regular em 2 tempos, bulhas normofonéticas, sem sopros."
    - "otoscopia: normal" DEVE ser expandido para "Otoscopia: Conduto auditivo externo pérvio, membrana timpânica íntegra, translúcida, sem abaulamentos ou retrações, com triângulo luminoso visível bilateralmente."
    - "oroscopia: normal" DEVE ser expandido para "Oroscopia: Mucosas úmidas e coradas, orofaringe sem hiperemia ou exsudatos, amígdalas eutróficas."
    - "abdome: normal" DEVE ser expandido para "Abdome: Plano, flácido, indolor à palpação, ruídos hidroaéreos presentes e normoativos, sem visceromegalias."
    - "neurológico: normal" DEVE ser expandido para "Neurológico: Vigil, orientado em tempo e espaço. Força, tônus e reflexos preservados. Pares cranianos sem alterações. Sensibilidade e coordenação normais. Marcha atípica."
    - **Generalize essa regra** para outros sistemas (pele, extremidades, etc.) quando a anotação for "[sistema]: normal".

---
MODELO E INSTRUÇÕES
---
`;

const TEMPLATES = {
    "adult-routine": {
        name: 'Rotina do Adulto',
        placeholder: 'Ex: Mulher, 45 anos, consulta de rotina, assintomática...',
        template: `## Antecedentes:
- Doenças prévias relevantes (listar cada doença ou condição mencionada, incluindo data de diagnóstico se disponível; omitir se não mencionado)
- Cirurgias prévias (listar cada procedimento, incluindo data se disponível; omitir se não mencionado)
- Alergias (listar substâncias e reações, se presentes; omitir se não mencionado)
- MUC (listar nome, dose e posologia; omitir se não mencionado)
- Histórico familiar de doenças (listar doenças e parentes afetados; omitir se não mencionado)
- Vacinação (listar vacinas recebidas ou faltantes; omitir se não mencionado)
- Outros antecedentes relevantes (ex: transfusões, hospitalizações, hábitos de vida como tabagismo, etilismo, uso de drogas, atividade física, alimentação, sono; listar apenas se mencionado)

## Subjetivo:
S1) Queixa principal do paciente, com descrição detalhada dos sintomas, início, duração, fatores de melhora e piora, tratamentos prévios e impacto na vida diária. (sempre incluir este item)
S2) Outras queixas ou sintomas não relacionados diretamente à queixa principal, de sistemas distintos, cada um em item separado, com descrição detalhada conforme acima. (incluir apenas se houver sintomas adicionais não relacionados; omitir se não mencionado)
S3), S4), ... (seguir a mesma lógica para sintomas adicionais de outros sistemas, se presentes)
Formato: cada item em lista, com texto corrido detalhado.

## Objetivo:
- Achados do exame físico geral (listar sinais vitais, estado geral, aparência, estado mental, alterações de pele e anexos, se disponíveis)
- Achados do exame físico por sistemas (listar achados relevantes para cada sistema examinado, ex: cardiovascular, respiratório, abdominal, neurológico, etc.; omitir sistemas não mencionados)
- Dados antropométricos (peso, altura, IMC, circunferência abdominal, se disponíveis)
- Resultados de exames complementares (listar nome do exame, data e resultados; omitir se não mencionado)
Formato: lista

## Avaliação:
- Lista dos problemas ativos identificados na consulta (cada problema em um item separado, incluindo sintomas, diagnósticos confirmados ou hipóteses diagnósticas, com código CID-10 se disponível)
Formato: lista

## Plano:
- Medicações prescritas (nome, dose, posologia; omitir se não mencionado)
- Orientações fornecidas ao paciente (listar cada orientação em item separado; omitir se não mencionado)
- Exames complementares solicitados (listar nome do exame; omitir se não mencionado)
- Encaminhamentos para outros profissionais ou serviços (listar especialidade ou serviço; omitir se não mencionado)
- Atestados ou documentos fornecidos (descrever tipo e duração; omitir se não mencionado)
Formato: lista`
    },
    "pediatrics": {
        name: 'Puericultura',
        placeholder: 'Ex: Lactente, 6 meses, consulta de rotina...',
        template: `## Antecedentes:
- Gestação: informações sobre a gestação, incluindo intercorrências, uso de medicações, doenças maternas, exames realizados, tipo de parto, idade gestacional ao nascimento (incluir apenas se mencionado explicitamente na transcrição ou nas anotações. Caso contrário, omitir completamente)
- Parto: tipo de parto, condições ao nascimento, Apgar, peso, comprimento, necessidade de reanimação (incluir apenas se mencionado explicitamente na transcrição ou nas anotações. Caso contrário, omitir completamente)
- Aleitamento: tipo de aleitamento (materno exclusivo, misto, artificial), tempo de aleitamento, dificuldades, introdução alimentar (incluir apenas se mencionado explicitamente na transcrição ou nas anotações. Caso contrário, omitir completamente)
- Vacinação: vacinas recebidas e pendentes, reações adversas (incluir apenas se mencionado explicitamente na transcrição ou nas anotações. Caso contrário, omitir completamente)
- Doenças prévias: doenças, hospitalizações, cirurgias, alergias, MUC (incluir apenas se mencionado explicitamente na transcrição ou nas anotações. Caso contrário, omitir completamente)
- História familiar: doenças genéticas, crônicas ou relevantes em familiares (incluir apenas se mencionado explicitamente na transcrição ou nas anotações. Caso contrário, omitir completamente)
- Desenvolvimento neuropsicomotor: marcos do desenvolvimento, atrasos, regressões (incluir apenas se mencionado explicitamente na transcrição ou nas anotações. Caso contrário, omitir completamente)

## Subjetivo:
- Queixa principal/motivo da consulta (incluir apenas se mencionado explicitamente na transcrição ou nas anotações. Caso contrário, omitir completamente)
- Sintomas atuais, duração, fatores de melhora/piora, medicações utilizadas (incluir apenas se mencionado explicitamente na transcrição ou nas anotações. Caso contrário, omitir completamente)
- Hábitos de sono, alimentação, evacuação, micção, comportamento, rotina diária (incluir apenas se mencionado explicitamente na transcrição ou nas anotações. Caso contrário, omitir completamente)
- Queixas emocionais, sociais ou comportamentais (incluir apenas se mencionado explicitamente na transcrição ou nas anotações. Caso contrário, omitir completamente)

## Objetivo:
- Dados antropométricos: peso, altura/comprimento, perímetro cefálico, IMC, percentis (incluir apenas se mencionado explicitamente na transcrição ou nas anotações. Caso contrário, omitir completamente)
- Sinais vitais: temperatura, frequência cardíaca, frequência respiratória, pressão arterial, saturação de O2 (incluir apenas se mencionado explicitamente na transcrição ou nas anotações. Caso contrário, omitir completamente)
- Exame físico geral: estado geral, hidratação, coloração, presença de lesões, alterações em pele e mucosas (incluir apenas se mencionado explicitamente na transcrição ou nas anotações. Caso contrário, omitir completamente)
- Exame físico segmentar: cabeça, olhos, ouvidos, nariz, garganta, tórax, abdome, genitália, membros, coluna, sistema neurológico, desenvolvimento motor (incluir apenas se mencionado explicitamente na transcrição ou nas anotações. Caso contrário, omitir completamente)
- Resultados de exames complementares: exames laboratoriais, de imagem ou outros, com data, nome do exame e valores (incluir apenas se mencionado explicitamente na transcrição ou nas anotações. Caso contrário, omitir completamente)

## Avaliação:
- Lista de problemas ativos identificados na consulta (incluir apenas se mencionado explicitamente na transcrição ou nas anotações. Caso contrário, omitir completamente)
- Diagnósticos confirmados e hipóteses diagnósticas, preferencialmente com código CID-10: descrição do código (incluir apenas se mencionado explicitamente na transcrição ou nas anotações. Caso contrário, omitir completamente)
- Situação do crescimento e desenvolvimento: adequado, em risco, atrasado, regressão (incluir apenas se mencionado explicitamente na transcrição ou nas anotações. Caso contrário, omitir completamente)

## Plano:
- Medicações prescritas: nome, dosagem, posologia (incluir apenas se mencionado explicitamente na transcrição ou nas anotações. Caso contrário, omitir completamente)
- Orientações fornecidas: alimentação, sono, higiene, prevenção de acidentes, estímulo ao desenvolvimento, vacinação, entre outros (incluir apenas se mencionado explicitamente na transcrição ou nas anotações. Caso contrário, omitir completamente)
- Exames complementares solicitados (incluir apenas se mencionado explicitamente na transcrição ou nas anotações. Caso contrário, omitir completamente)
- Encaminhamentos para outras especialidades ou serviços (incluir apenas se mencionado explicitamente na transcrição ou nas anotações. Caso contrário, omitir completamente)
- Retorno: prazo e motivo do retorno agendado (incluir apenas se mencionado explicitamente na transcrição ou nas anotações. Caso contrário, omitir completamente)`
    },
    "prenatal": {
        name: 'Pré-natal',
        placeholder: 'Ex: Gestante, 28 anos, G2P1A0, IG 28 semanas...',
        template: `## Antecedentes:
- Idade gestacional atual (em semanas) (preencher apenas se mencionado explicitamente na transcrição; caso contrário, omitir)
- Gestações anteriores: número de gestações, partos, abortos, cesáreas (preencher apenas se mencionado explicitamente na transcrição; caso contrário, omitir)
- Doenças prévias maternas (ex: hipertensão, diabetes, doenças autoimunes, etc.) (preencher apenas se mencionado explicitamente na transcrição; caso contrário, omitir)
- Cirurgias prévias (preencher apenas se mencionado explicitamente na transcrição; caso contrário, omitir)
- Alergias (preencher apenas se mencionado explicitamente na transcrição; caso contrário, omitir)
- MUC (preencher apenas se mencionado explicitamente na transcrição; caso contrário, omitir)
- História familiar de doenças relevantes (preencher apenas se mencionado explicitamente na transcrição; caso contrário, omitir)
- História obstétrica relevante (ex: pré-eclâmpsia, prematuridade, restrição de crescimento, etc.) (preencher apenas se mencionado explicitamente na transcrição; caso contrário, omitir)
Formato: lista

## Subjetivo:
Descrever queixa principal, sintomas atuais, desconfortos, dúvidas, alterações percebidas pela gestante, evolução da gestação, intercorrências desde a última consulta, adesão ao pré-natal, estado emocional, hábitos de vida (alimentação, atividade física, uso de substâncias), queixas urinárias, digestivas, mamárias, entre outras. (incluir apenas o que foi mencionado explicitamente na transcrição; caso contrário, omitir completamente)
Formato: texto corrido

## Objetivo:
- Sinais vitais: pressão arterial, frequência cardíaca, temperatura, frequência respiratória (preencher apenas se mencionado explicitamente na transcrição; caso contrário, omitir)
- Peso atual (preencher apenas se mencionado explicitamente na transcrição; caso contrário, omitir)
- Altura uterina (preencher apenas se mencionado explicitamente na transcrição; caso contrário, omitir)
- Batimentos cardíacos fetais (preencher apenas se mencionado explicitamente na transcrição; caso contrário, omitir)
- Movimentação fetal (preencher apenas se mencionado explicitamente na transcrição; caso contrário, omitir)
- Exame físico geral (ex: edema, alterações mamárias, alterações cutâneas, etc.) (preencher apenas se mencionado explicitamente na transcrição; caso contrário, omitir)
- Exames laboratoriais e de imagem realizados, com data, nome do exame e resultados (preencher apenas se mencionado explicitamente na transcrição; caso contrário, omitir)
Formato: lista

## Avaliação:
- Diagnóstico gestacional (ex: gestação única, gemelar, risco habitual, alto risco, etc.) (preencher apenas se mencionado explicitamente na transcrição; caso contrário, omitir)
- Problemas ou complicações identificadas (ex: anemia, infecção urinária, diabetes gestacional, hipertensão, etc.) (preencher apenas se mencionado explicitamente na transcrição; caso contrário, omitir)
- Hipóteses diagnósticas e diagnósticos confirmados, preferencialmente com código CID-10 (preencher apenas se mencionado explicitamente na transcrição; caso contrário, omitir)
Formato: lista

## Plano:
- Medicações prescritas, com nome, dosagem e posologia (preencher apenas se mencionado explicitamente na transcrição; caso contrário, omitir)
- Suplementações orientadas (ex: ácido fólico, ferro, polivitamínicos) (preencher apenas se mencionado explicitamente na transcrição; caso contrário, omitir)
- Exames complementares solicitados (preencher apenas se mencionado explicitamente na transcrição; caso contrário, omitir)
- Orientações fornecidas à gestante (ex: sinais de alarme, alimentação, atividade física, retorno) (preencher apenas se mencionado explicitamente na transcrição; caso contrário, omitir)
- Encaminhamentos realizados (preencher apenas se mencionado explicitamente na transcrição; caso contrário, omitir)
- Agendamento de retorno (preencher apenas se mencionado explicitamente na transcrição; caso contrário, omitir)
Formato: lista`
    },
    "mental-health": {
        name: 'Saúde Mental',
        placeholder: 'Ex: Paciente 34a, refere tristeza profunda...',
        template: `## Queixa Principal:
Queixa do paciente/motivo da consulta.
Formato: Texto corrido

## História da Doença Atual:
História da doença atual, com sintomas, duração dos sintomas, fatores de melhora e piora, medicações utilizadas para os sintomas, entre outros. Incluir queixas, sintomas e informações sobre o estado emocional/psicológico, caso estejam presentes. Essa deve ser a parte mais rica e completa de todo o registro, trazendo todos os pontos relevantes para o estado mental do paciente, de forma detalhada. Todos os sintomas devem ser mencionados e detalhados.
Aqui também devem constar os relatos do paciente sobre suas aflições, histórias ocorridas recentemente, anseios, medos e expectativas, da forma mais detalhada possível.
Formato: Texto corrido

## História Médica e Psiquiátrica:
Histórico pessoal de condições médicas e psiquiátricas, incluindo histórico de tratamento psiquiátrico, hospitalizações, alergias a medicamentos e outras questões médicas relevantes. Incluir também histórico de traumatismos e cirurgias. Incluir marcos do desenvolvimento neuropsicomotor e situações de vida marcantes para o contexto do paciente.
Formato: Texto corrido

## Tratamento:
Descrição de tratamento atual e/ou recente, incluindo terapia e medicações, com informações sobre a eficácia e os efeitos colaterais. Incluir tanto tratamentos farmacológicos quanto não farmacológicos.
Indicar com clareza quais medicamentos o paciente está usando atualmente, com nome, dosagem e posologias.
Utilize nomes de medicamentos reais e válidos em português.
Formato: lista

## Hábitos de Vida:
Informações de hábitos de vida do paciente: padrão de sono, de atividade física, de alimentação e de atividades de lazer. Informar sobre uso de tabaco, álcool e outras drogas, incluindo padrão de consumo.
Formato: Texto corrido

## História Social:
Informações sobre ambiente familiar, nível educacional, trabalho, relacionamentos, eventos traumáticos passados e experiências significativas.
Formato: Texto corrido

## História Familiar:
Histórico de doenças mentais e outras condições médicas em familiares.
Formato: Texto corrido

## Exame do Estado Mental:
Achados do exame mental, incluindo aparência e comportamento, atitude, fala, conteúdo do pensamento, forma do pensamento, percepção, afeto e humor, função cognitiva, nível de consciência, orientação e avaliação de risco.
Formato: Texto corrido

## Dados Antropométricos: responder apenas se disponíveis.
- Peso: Peso do paciente em kg, se disponível (ex: 75 kg)
- Altura: Altura do paciente em cm, se disponível (ex: 176 cm)
- IMC: peso em kg dividida pelo quadrado da altura em metros, se peso e altura disponíveis (ex: 24,6 Kg/m2)

## Resultado de Exames:
Inserir aqui todos os resultados de exames complementares, com data, nome do exame e valores.
Formato: lista

## Hipóteses Diagnósticas:
Diagnósticos confirmados e hipóteses diagnósticas, usando código CID-10, inclusive com base nas medicações em uso. Forma da resposta: código CID-10: descrição do código
Formato: lista

## Medicações Prescritas:
Medicações que foram prescritas, com nome, dosagem e posologia.
Formato: lista

## Orientações:
Orientações dadas ao paciente.
Formato: lista

## Exames Complementares:
Exames complementares que foram solicitados para o paciente realizar.
Formato: lista

## Atestado:
Atestado médico com número de dias de afastamento que foi prescrito ao paciente.
Formato: texto corrido

## Encaminhamento:
Caso o médico explicitamente encaminhou o paciente para avaliação ou acompanhamento com outro especialista ou profissional de saúde, descrever aqui fielmente ao que consta em [TRANSCRIÇÃO] ou em [ANOTAÇÕES]. Lembre-se: é uma falha grave citar encaminhamentos que não constam explicitamente em [TRANSCRIÇÃO] ou em [ANOTAÇÕES]. Simplesmente oculte essa seção se não tiver informações para preenchê-la. Na dúvida, oculte a seção. Não incluir encaminhamento para realização de exames.
Formato: texto corrido`
    },
};

// --- DOM ELEMENTS ---
const promptInput = document.getElementById('prompt-input') as HTMLTextAreaElement;
const generateButton = document.getElementById('generate-button') as HTMLButtonElement;
const outputContent = document.getElementById('output-content') as HTMLDivElement;
const loader = document.getElementById('loader') as HTMLDivElement;
const templateTabs = document.getElementById('template-tabs') as HTMLDivElement;

// --- UI UPDATES ---

function updateUIForActiveTemplate() {
    const activeTemplate = TEMPLATES[activeTemplateId as keyof typeof TEMPLATES];
    if (activeTemplate) {
        promptInput.placeholder = activeTemplate.placeholder;
        outputContent.innerHTML = `<p class="placeholder">A anamnese organizada para <strong>${activeTemplate.name}</strong> aparecerá aqui.</p>`;
    }

    // Update active tab style
    const tabs = templateTabs.querySelectorAll('.tab');
    tabs.forEach(tab => {
        if ((tab as HTMLElement).dataset.template === activeTemplateId) {
            tab.classList.add('active');
        } else {
            tab.classList.remove('active');
        }
    });
}

// --- API & GENERATION ---

function formatResponse(text: string): string {
    return text
        .replace(/^## (.*?)$/gm, '<h3>$1</h3>')
        .split('\n')
        .map(line => {
            const trimmedLine = line.trim();
            if (trimmedLine.startsWith('- ')) {
                return `<li>${trimmedLine.substring(2)}</li>`;
            } else if (trimmedLine.length > 0 && !trimmedLine.startsWith('<h3')) {
                return `<p>${trimmedLine}</p>`;
            }
            return trimmedLine;
        })
        .join('')
        .replace(/<\/h3><li>/g, '</h3><ul><li>')
        .replace(/<li>(?!<\/li>)/g, '<li>')
        .replace(/(<\/li>)(?!<li>)/g, '$1</ul>');
}

function setGeneratingState(generating: boolean) {
    isGenerating = generating;
    if (generating) {
        generateButton.textContent = 'Parar Geração';
        generateButton.classList.add('stop-button');
        loader.classList.remove('hidden');
        outputContent.classList.add('hidden');
    } else {
        generateButton.textContent = 'Organizar Anamnese';
        generateButton.classList.remove('stop-button');
        generateButton.disabled = false;
        loader.classList.add('hidden');
        outputContent.classList.remove('hidden');
        abortController = null;
    }
}

async function generateAnamnesis() {
    if (!promptInput.value.trim()) {
        outputContent.innerHTML = '<p class="placeholder">Por favor, insira as anotações do paciente.</p>';
        return;
    }

    const activeTemplate = TEMPLATES[activeTemplateId as keyof typeof TEMPLATES];
    if (!activeTemplate) {
        outputContent.innerHTML = '<p class="placeholder" style="color: red;">Nenhum modelo selecionado.</p>';
        return;
    }

    setGeneratingState(true);
    generateButton.disabled = false; // Keep it enabled to act as a stop button

    abortController = new AbortController();
    const userInput = promptInput.value;
    const fullPrompt = `${BASE_INSTRUCTION}${activeTemplate.template}\n\nAnotações do Profissional:\n${userInput}`;

    try {
        const request = {
            model: "gemini-2.5-flash",
            contents: fullPrompt,
        };

        const response = await ai.models.generateContent(request);

        if (abortController?.signal.aborted) {
            console.log("Generation was cancelled, ignoring response.");
            return;
        }

        const anamnesisText = response.text;
        outputContent.innerHTML = formatResponse(anamnesisText);
    } catch (error) {
        if ((error as Error).name === 'AbortError') {
             outputContent.innerHTML = `<p class="placeholder">Geração cancelada pelo usuário.</p>`;
             console.log("Generation cancelled by user.");
        } else {
            console.error("Error generating anamnesis:", error);
            outputContent.innerHTML = `<p class="placeholder" style="color: red;">Ocorreu um erro ao gerar a anamnese. Por favor, tente novamente.</p>`;
        }
    } finally {
        setGeneratingState(false);
    }
}

function handleGenerateClick() {
    if (isGenerating) {
        abortController?.abort();
        setGeneratingState(false);
        outputContent.innerHTML = `<p class="placeholder">Geração cancelada pelo usuário.</p>`;
    } else {
        generateAnamnesis();
    }
}

// --- INITIALIZATION ---
function initialize() {
    // Event Listeners
    generateButton.addEventListener('click', handleGenerateClick);
    
    const tabs = templateTabs.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const templateId = (tab as HTMLElement).dataset.template;
            if (templateId) {
                activeTemplateId = templateId;
                promptInput.value = '';
                updateUIForActiveTemplate();
            }
        });
    });

    // Initial setup
    updateUIForActiveTemplate();
}

window.addEventListener('DOMContentLoaded', initialize);