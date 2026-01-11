package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
)

// QdrantStore реализует хранение векторов в Qdrant через REST API
type QdrantStore struct {
	baseURL    string
	collection string
	embedder   embeddings.Embedder
	client     *http.Client
}

// NewQdrantStore создает новое подключение к Qdrant
func NewQdrantStore(ctx context.Context, baseURL, collection string, embedder embeddings.Embedder) (*QdrantStore, error) {
	store := &QdrantStore{
		baseURL:    baseURL,
		collection: collection,
		embedder:   embedder,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}

	// Создаем коллекцию если она не существует
	err := store.createCollectionIfNotExists(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to create collection: %w", err)
	}

	return store, nil
}

// createCollectionIfNotExists создает коллекцию если она не существует
func (s *QdrantStore) createCollectionIfNotExists(ctx context.Context) error {
	// Проверяем размерность вектора, создавая тестовый эмбеддинг
	testEmbedding, err := s.embedder.EmbedQuery(ctx, "test")
	if err != nil {
		return fmt.Errorf("failed to create test embedding: %w", err)
	}

	vectorSize := len(testEmbedding)

	// Создаем коллекцию
	collectionConfig := map[string]interface{}{
		"vectors": map[string]interface{}{
			"size":     vectorSize,
			"distance": "Cosine",
		},
	}

	body, _ := json.Marshal(collectionConfig)
	req, err := http.NewRequestWithContext(ctx, "PUT", s.baseURL+"/collections/"+s.collection, bytes.NewBuffer(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// Если коллекция уже существует, игнорируем ошибку
	if resp.StatusCode != 200 && resp.StatusCode != 409 {
		return fmt.Errorf("failed to create collection: status %d", resp.StatusCode)
	}

	return nil
}

// AddDocuments добавляет документы в Qdrant
func (s *QdrantStore) AddDocuments(ctx context.Context, docs []schema.Document, _ ...vectorstores.Option) ([]string, error) {
	var ids []string

	for _, doc := range docs {
		// Создаем эмбеддинг для документа
		vector, err := s.embedder.EmbedQuery(ctx, doc.PageContent)
		if err != nil {
			return nil, fmt.Errorf("failed to embed document: %w", err)
		}

		// Генерируем уникальный ID
		id := uuid.New().String()
		ids = append(ids, id)

		// Создаем payload с метаданными
		payload := map[string]interface{}{
			"content": doc.PageContent,
		}
		for key, value := range doc.Metadata {
			payload[key] = value
		}

		// Добавляем точку в Qdrant
		point := map[string]interface{}{
			"id":      id,
			"vector":  vector,
			"payload": payload,
		}

		body, _ := json.Marshal(map[string]interface{}{
			"points": []interface{}{point},
		})

		req, err := http.NewRequestWithContext(ctx, "PUT", s.baseURL+"/collections/"+s.collection+"/points", bytes.NewBuffer(body))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := s.client.Do(req)
		if err != nil {
			return nil, err
		}
		defer resp.Body.Close()

		if resp.StatusCode != 200 {
			return nil, fmt.Errorf("failed to upsert point: status %d", resp.StatusCode)
		}
	}

	return ids, nil
}

// SimilaritySearch выполняет поиск похожих документов
func (s *QdrantStore) SimilaritySearch(ctx context.Context, query string, numDocs int, _ ...vectorstores.Option) ([]schema.Document, error) {
	// Создаем эмбеддинг для запроса
	vector, err := s.embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %w", err)
	}

	// Выполняем поиск
	searchRequest := map[string]interface{}{
		"vector":       vector,
		"limit":        numDocs,
		"with_payload": true,
	}

	body, _ := json.Marshal(searchRequest)
	req, err := http.NewRequestWithContext(ctx, "POST", s.baseURL+"/collections/"+s.collection+"/points/search", bytes.NewBuffer(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("failed to search: status %d", resp.StatusCode)
	}

	// Парсим ответ
	var searchResult struct {
		Result []struct {
			ID      string                 `json:"id"`
			Score   float64                `json:"score"`
			Payload map[string]interface{} `json:"payload"`
		} `json:"result"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&searchResult); err != nil {
		return nil, fmt.Errorf("failed to decode search result: %w", err)
	}

	// Преобразуем результаты в документы
	var docs []schema.Document
	for _, result := range searchResult.Result {
		content, ok := result.Payload["content"].(string)
		if !ok {
			continue
		}

		metadata := make(map[string]any)
		for key, value := range result.Payload {
			if key != "content" {
				metadata[key] = value
			}
		}

		docs = append(docs, schema.Document{
			PageContent: content,
			Metadata:    metadata,
		})
	}

	return docs, nil
}

type LocalAILLM struct {
	baseURL string
	model   string
	client  *http.Client
}

func NewLocalAILLM(baseURL, model string) *LocalAILLM {
	return &LocalAILLM{
		baseURL: baseURL,
		model:   model,
		client: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

type chatReq struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResp struct {
	Choices []struct {
		Message chatMessage `json:"message"`
	} `json:"choices"`
}

// func (l *LocalAILLM) Predict(ctx context.Context, prompt string) (string, error) {
// 	reqBody := chatReq{
// 		Model: l.model,
// 		Messages: []chatMessage{
// 			{Role: "system", Content: "Ты помощник, отвечай кратко и по делу."},
// 			{Role: "user", Content: prompt},
// 		},
// 	}
// 	b, _ := json.Marshal(reqBody)

// 	req, err := http.NewRequestWithContext(
// 		ctx,
// 		http.MethodPost,
// 		l.baseURL+"/v1/chat/completions",
// 		bytes.NewReader(b),
// 	)
// 	if err != nil {
// 		return "", err
// 	}
// 	req.Header.Set("Content-Type", "application/json")

// 	resp, err := l.client.Do(req)
// 	if err != nil {
// 		return "", err
// 	}
// 	defer resp.Body.Close()

// 	if resp.StatusCode != http.StatusOK {
// 		return "", fmt.Errorf("localai chat status: %d", resp.StatusCode)
// 	}

// 	var r chatResp
// 	if err := json.NewDecoder(resp.Body).Decode(&r); err != nil {
// 		return "", err
// 	}
// 	if len(r.Choices) == 0 {
// 		return "", fmt.Errorf("no choices returned")
// 	}
// 	return r.Choices[0].Message.Content, nil
// }

func (l *LocalAILLM) PredictStream(ctx context.Context, prompt string, onChunk func(string)) error {
	reqBody := chatReq{
		Model: l.model,
		Messages: []chatMessage{
			{Role: "system", Content: "Ты краткий ассистент."},
			{Role: "user", Content: prompt},
		},
		Stream: true,
	}

	b, _ := json.Marshal(reqBody)
	req, _ := http.NewRequestWithContext(ctx, "POST", l.baseURL+"/v1/chat/completions", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")

	resp, err := l.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// Используем сканер для чтения потока SSE по строкам
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" || line == "data: [DONE]" {
			continue
		}

		// Убираем префикс "data: "
		line = strings.TrimPrefix(line, "data: ")

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}

		if err := json.Unmarshal([]byte(line), &chunk); err == nil {
			if len(chunk.Choices) > 0 {
				content := chunk.Choices[0].Delta.Content
				onChunk(content) // Передаем кусочек текста в коллбэк
			}
		}
	}
	return scanner.Err()
}

func GetData(ctx context.Context, llm *LocalAILLM, store *QdrantStore, input string) {
	docs, _ := store.SimilaritySearch(ctx, input, 3)
	contextInfo := ""
	for _, d := range docs {
		contextInfo += d.PageContent + "\n"
	}

	prompt := fmt.Sprintf("Контекст:\n%s\nВопрос: %s", contextInfo, input)

	fmt.Print("Ответ: ")
	var fullResponse strings.Builder

	// 2. Стриминг ответа
	err := llm.PredictStream(ctx, prompt, func(chunk string) {
		fmt.Print(chunk)                // Печатаем в консоль сразу
		fullResponse.WriteString(chunk) // Накапливаем полный ответ
	})
	fmt.Println() // Перенос строки в конце

	if err != nil {
		log.Printf("Stream error: %v", err)
		return
	}

	// 3. Сохранение (используем накопленный fullResponse)
	store.AddDocuments(ctx, []schema.Document{{
		PageContent: fmt.Sprintf("Q: %s\nA: %s", input, fullResponse.String()),
	}})
}

func main() {

	ctx := context.Background()

	// LocalAI клиенты
	llm := NewLocalAILLM("http://localhost:8080", "local-llm") // имя модели из YAML
	embedder := NewLocalAIEmbedder("http://localhost:8080", "local-embed")

	// Qdrant
	qdrantStore, err := NewQdrantStore(ctx, "http://localhost:6333", "documents", embedder)
	if err != nil {
		log.Fatalf("Ошибка подключения к Qdrant: %v", err)
	}

	// 4. Добавляем начальные документы
	fmt.Println("Добавление начальных документов...")
	initialDocs := []schema.Document{
		{PageContent: "Ollama - это инструмент для запуска локальных LLM моделей."},
		{PageContent: "Qdrant - это векторная база данных для хранения и поиска векторов."},
		{PageContent: "RAG (Retrieval-Augmented Generation) - это подход, объединяющий поиск информации и генерацию текста."},
		{PageContent: "Go - это язык программирования, разработанный Google."},
	}

	_, err = qdrantStore.AddDocuments(ctx, initialDocs)
	if err != nil {
		log.Printf("Ошибка добавления документов: %v", err)
	} else {
		fmt.Println("Документы успешно добавлены в Qdrant")
	}

	fmt.Println("\n=== RAG система с LocalAI и Qdrant ===")
	fmt.Println("Введите ваш вопрос (или 'exit' для выхода):")

	// Создаем ридер для стандартного ввода
	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Print("\n> ")

		// Читаем всю строку целиком
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Printf("Ошибка чтения: %v", err)
			continue
		}

		input = strings.TrimSpace(input)

		if input == "exit" {
			fmt.Println("Завершение работы...")
			break
		}

		if input == "" {
			continue
		}

		GetData(ctx, llm, qdrantStore, input)
	}
}

type LocalAIEmbedder struct {
	baseURL string
	model   string
	client  *http.Client
}

func NewLocalAIEmbedder(baseURL, model string) embeddings.Embedder {
	return &LocalAIEmbedder{
		baseURL: baseURL,
		model:   model,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

type localAIEmbReq struct {
	Model string      `json:"model"`
	Input interface{} `json:"input"` // строка или []string
}

type localAIEmbResp struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
}

func (e *LocalAIEmbedder) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	reqBody := localAIEmbReq{
		Model: e.model,
		Input: text,
	}
	b, _ := json.Marshal(reqBody)

	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		e.baseURL+"/v1/embeddings",
		bytes.NewReader(b),
	)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("localai embeddings status: %d", resp.StatusCode)
	}

	var r localAIEmbResp
	if err := json.NewDecoder(resp.Body).Decode(&r); err != nil {
		return nil, err
	}
	if len(r.Data) == 0 {
		return nil, fmt.Errorf("no embedding returned")
	}

	// конвертируем []float64 -> []float32
	embF32 := make([]float32, len(r.Data[0].Embedding))
	for i, v := range r.Data[0].Embedding {
		embF32[i] = float32(v)
	}
	return embF32, nil
}

func (e *LocalAIEmbedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error) {
	vectors := make([][]float32, len(texts))
	for i, t := range texts {
		v, err := e.EmbedQuery(ctx, t)
		if err != nil {
			return nil, err
		}
		vectors[i] = v
	}
	return vectors, nil
}
