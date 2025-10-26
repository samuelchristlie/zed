use anyhow::Result;
use futures::{FutureExt, StreamExt, AsyncReadExt, future::{self, BoxFuture}};
use gpui::{AnyView, App, AsyncApp, Context, Entity, FocusHandle, Focusable, SharedString, Task, Window};
use http_client::{HttpClient, Method, AsyncBody, Request as HttpRequest};
use language_model::{
    AuthenticateError, LanguageModel, LanguageModelCompletionError, LanguageModelCompletionEvent,
    LanguageModelId, LanguageModelName, LanguageModelProvider, LanguageModelProviderId,
    LanguageModelProviderName, LanguageModelProviderState, LanguageModelRequest,
    LanguageModelToolChoice, MessageContent, RateLimiter, StopReason, TokenUsage,
    LanguageModelToolUseId, LanguageModelToolUse,
};
use serde::{Deserialize, Serialize};
use settings::{Settings, SettingsStore};
use std::pin::Pin;

use std::sync::{Arc, LazyLock};
use strum::{EnumIter, IntoEnumIterator};
use ui::{List, prelude::*};
use menu;
use ui_input::InputField;
use util::ResultExt;
use zed_env_vars::{EnvVar, env_var};

use crate::api_key::ApiKeyState;

// Z.AI API Structures
#[derive(Debug, Serialize, Deserialize)]
pub struct ChatThinking {
    pub r#type: ChatThinkingType,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatThinkingType {
    Enabled,
    Disabled,
}

#[derive(Debug, Serialize)]
pub struct ZAiRequest {
    pub model: String,
    pub messages: Vec<ZAiMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ChatThinking>,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ZAiTool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ZAiToolChoice>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
}

fn default_temperature() -> f32 {
    1.0
}

#[derive(Debug, Serialize)]
pub struct ZAiMessage {
    pub role: ZAiRole,
    pub content: ZAiContent,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ZAiRole {
    User,
    Assistant,
    System,
    Tool,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ZAiContent {
    Text(String),
    ToolCalls(Vec<ZAiToolCall>),
    ToolResult(String),
}

#[derive(Debug, Serialize)]
pub struct ZAiTool {
    pub r#type: String,
    pub function: ZAiFunction,
}

#[derive(Debug, Serialize)]
pub struct ZAiFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ZAiToolChoice {
    Auto,
    None,
}

#[derive(Debug, Serialize)]
pub struct ZAiToolCall {
    pub id: String,
    pub r#type: String,
    pub function: ZAiFunctionCall,
}

#[derive(Debug, Serialize)]
pub struct ZAiFunctionCall {
    pub name: String,
    pub arguments: String,
}

pub fn into_z_ai(
    request: LanguageModelRequest,
    model_id: &str,
    thinking_enabled: bool,
) -> ZAiRequest {
    let mut messages = Vec::new();

    for message in request.messages {
        if message.contents_empty() {
            continue;
        }

        let role = match message.role {
            language_model::Role::User => ZAiRole::User,
            language_model::Role::Assistant => ZAiRole::Assistant,
            language_model::Role::System => ZAiRole::System,
        };

        let mut message_parts = Vec::new();
        for content_item in message.content {
            match content_item {
                MessageContent::Text(text) => {
                    if !text.is_empty() {
                        message_parts.push(text);
                    }
                }
                MessageContent::Thinking { text, .. } => {
                    if !text.is_empty() {
                        message_parts.push(text);
                    }
                }
                MessageContent::RedactedThinking(_) => {
                }
                MessageContent::Image(_) => {
                }
                MessageContent::ToolUse(_) => {
                }
                MessageContent::ToolResult(_) => {
                }
            }
        }

        let content = message_parts.join(" ");
        if !content.is_empty() {
            messages.push(ZAiMessage {
                role,
                content: ZAiContent::Text(content),
            });
        }
    }

    let tools: Vec<ZAiTool> = request.tools.into_iter().map(|tool| ZAiTool {
        r#type: "function".to_string(),
        function: ZAiFunction {
            name: tool.name,
            description: tool.description,
            parameters: tool.input_schema,
        },
    }).collect();

    let tool_choice = request.tool_choice.map(|choice| match choice {
        LanguageModelToolChoice::Auto => ZAiToolChoice::Auto,
        LanguageModelToolChoice::None => ZAiToolChoice::None,
        _ => ZAiToolChoice::Auto,
    });

    ZAiRequest {
        model: model_id.to_string(),
        messages,
        thinking: if thinking_enabled {
            Some(ChatThinking {
                r#type: ChatThinkingType::Enabled,
            })
        } else {
            None
        },
        temperature: request.temperature.unwrap_or(1.0),
        max_tokens: None,
        stream: false,
        tools,
        tool_choice,
        stop: request.stop,
    }
}

const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("z_ai");
const PROVIDER_NAME: LanguageModelProviderName = LanguageModelProviderName::new("Z.AI");
const Z_AI_API_URL: &str = "https://api.z.ai/api/coding/paas/v4";
const API_KEY_ENV_VAR_NAME: &str = "Z_AI_API_KEY";
static API_KEY_ENV_VAR: LazyLock<EnvVar> = env_var!(API_KEY_ENV_VAR_NAME);

#[derive(Default, Clone, Debug, PartialEq)]
pub struct ZAiSettings {
    pub api_url: String,
    pub available_models: Vec<ZAiAvailableModel>,
}

pub struct ZAiLanguageModelProvider {
    http_client: Arc<dyn HttpClient>,
    state: Entity<State>,
}

pub struct State {
    api_key_state: ApiKeyState,
}

impl State {
    fn is_authenticated(&self) -> bool {
        self.api_key_state.has_key()
    }

    fn set_api_key(&mut self, api_key: Option<String>, cx: &mut Context<Self>) -> Task<Result<()>> {
        let api_url = ZAiLanguageModelProvider::api_url(cx);
        self.api_key_state
            .store(api_url, api_key, |this| &mut this.api_key_state, cx)
    }

    fn authenticate(&mut self, cx: &mut Context<Self>) -> Task<Result<(), AuthenticateError>> {
        let api_url = ZAiLanguageModelProvider::api_url(cx);
        self.api_key_state.load_if_needed(
            api_url,
            &API_KEY_ENV_VAR,
            |this| &mut this.api_key_state,
            cx,
        )
    }
}

impl ZAiLanguageModelProvider {
    pub fn new(http_client: Arc<dyn HttpClient>, cx: &mut App) -> Self {
        let state = cx.new(|cx| {
            cx.observe_global::<SettingsStore>(|this: &mut State, cx| {
                let api_url = Self::api_url(cx);
                this.api_key_state.handle_url_change(
                    api_url,
                    &API_KEY_ENV_VAR,
                    |this| &mut this.api_key_state,
                    cx,
                );
                cx.notify();
            })
            .detach();
            State {
                api_key_state: ApiKeyState::new(Self::api_url(cx)),
            }
        });

        Self { http_client, state }
    }

    fn api_url(cx: &App) -> SharedString {
        let api_url = &Self::settings(cx).api_url;
        if api_url.is_empty() {
            Z_AI_API_URL.into()
        } else {
            SharedString::new(api_url.as_str())
        }
    }

    fn create_language_model(&self, model: ZAiModel) -> Arc<dyn LanguageModel> {
        Arc::new(ZAiLanguageModel {
            id: LanguageModelId::from(model.id().to_string()),
            model,
            state: self.state.clone(),
            http_client: self.http_client.clone(),
            request_limiter: RateLimiter::new(4),
        })
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        let settings = ZAiLanguageModelProvider::settings(cx);
        let mut models = Vec::new();

        // Add default Z.AI models if none are configured
        if settings.available_models.is_empty() {
            for model in ZAiModel::iter() {
                // Skip the Custom variant for default models
                if !matches!(model, ZAiModel::Custom { .. }) {
                    models.push(self.create_language_model(model));
                }
            }
        } else {
            // Add configured models
            for available_model in &settings.available_models {
                let model = ZAiModel::Custom {
                    name: available_model.name.clone(),
                    display_name: available_model.display_name.clone(),
                    max_tokens: available_model.max_tokens,
                    max_output_tokens: available_model.max_output_tokens,
                    supports_thinking: available_model.supports_thinking,
                };
                models.push(self.create_language_model(model));
            }
        }

        models
    }

    fn settings(cx: &App) -> ZAiSettings {
        crate::AllLanguageModelSettings::get_global(cx).z_ai.clone()
    }
}

impl LanguageModelProvider for ZAiLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn default_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(Arc::new(ZAiLanguageModel {
            id: LanguageModelId::from("glm-4.6".to_string()),
            model: ZAiModel::Glm46,
            state: self.state.clone(),
            http_client: self.http_client.clone(),
            request_limiter: RateLimiter::new(4),
        }))
    }

    fn default_fast_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(Arc::new(ZAiLanguageModel {
            id: LanguageModelId::from("glm-4.5-air".to_string()),
            model: ZAiModel::Glm45Air,
            state: self.state.clone(),
            http_client: self.http_client.clone(),
            request_limiter: RateLimiter::new(4),
        }))
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        self.provided_models(cx)
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        self.state.read(cx).is_authenticated()
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        self.state.update(cx, |state, cx| state.authenticate(cx))
    }

    fn configuration_view(
        &self,
        target_agent: language_model::ConfigurationViewTargetAgent,
        window: &mut Window,
        cx: &mut App,
    ) -> AnyView {
        cx.new(|cx| ZAiConfigurationView::new(self.state.clone(), target_agent, window, cx))
            .into()
    }

    fn reset_credentials(&self, cx: &mut App) -> Task<Result<()>> {
        self.state
            .update(cx, |state, cx| state.set_api_key(None, cx))
    }
}

impl LanguageModelProviderState for ZAiLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

pub struct ZAiLanguageModel {
    id: LanguageModelId,
    model: ZAiModel,
    state: Entity<State>,
    http_client: Arc<dyn HttpClient>,
    request_limiter: RateLimiter,
}

impl LanguageModel for ZAiLanguageModel {
    fn id(&self) -> LanguageModelId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelName {
        LanguageModelName::from(self.model.display_name().to_string())
    }

    fn provider_id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn provider_name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn telemetry_id(&self) -> String {
        format!("z_ai:{}", self.model.id())
    }

    fn max_output_tokens(&self) -> Option<u64> {
        self.model.max_output_tokens().map(|t| t as u64)
    }

    fn max_token_count(&self) -> u64 {
        self.model.max_tokens()
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_tool_choice(&self, _choice: LanguageModelToolChoice) -> bool {
        true
    }

    fn supports_images(&self) -> bool {
        false
    }

    fn count_tokens(
        &self,
        request: LanguageModelRequest,
        cx: &App,
    ) -> BoxFuture<'static, Result<u64>> {
        // For now, estimate using simple character-based tokenization
        // Z.AI models should ideally have their own tokenizer, but we'll use a rough estimate
        let total_chars: usize = request.messages.iter().map(|msg| {
            msg.content.iter().map(|content| {
                match content {
                    MessageContent::Text(text) => text.len(),
                    MessageContent::Thinking { text, .. } => text.len(),
                    MessageContent::Image(_) => 1000, // Rough estimate for images
                    MessageContent::ToolUse(tool_use) => tool_use.name.len() + tool_use.input.to_string().len(),
                    MessageContent::ToolResult(tool_result) => format!("{:?}", tool_result).len(),
                    MessageContent::RedactedThinking(data) => data.len(),
                }
            }).sum::<usize>()
        }).sum();

        Box::pin(cx.background_spawn(async move {
            // Rough estimate: 1 token ≈ 4 characters for most models
            Ok((total_chars / 4) as u64)
        }))
    }

    fn stream_completion(
        &self,
        request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            futures::stream::BoxStream<'static, Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>,
            LanguageModelCompletionError,
        >,
    > {
        let http_client = self.http_client.clone();
        let model = self.model.clone();
        let request_limiter = self.request_limiter.clone();
        
        // Get the API key and URL before entering the rate-limited async block
        let (api_key, api_url) = match self.state.read_with(cx, |state, cx| {
            let api_url = ZAiLanguageModelProvider::api_url(cx);
            (state.api_key_state.key(&api_url), api_url.to_string())
        }) {
            Ok((Some(api_key), api_url)) => (api_key, api_url),
            Ok((None, _)) => {
                return future::ready(Err(LanguageModelCompletionError::NoApiKey {
                    provider: PROVIDER_NAME,
                })).boxed();
            },
            Err(_) => {
                return future::ready(Err(LanguageModelCompletionError::NoApiKey {
                    provider: PROVIDER_NAME,
                })).boxed();
            }
        };

        let future = request_limiter.stream(async move {
            let thinking_enabled = request.thinking_allowed && model.supports_thinking();
            let z_ai_request = into_z_ai(request, model.id(), thinking_enabled);

            let api_url = api_url.trim_end_matches('/');

            let request_body = serde_json::to_string(&z_ai_request)
                .map_err(|error| LanguageModelCompletionError::from_cloud_failure(
                    PROVIDER_NAME,
                    "serialization_error".to_string(),
                    format!("Failed to serialize request: {}", error),
                    None,
                ))?;

            let http_request = HttpRequest::builder()
                .method(Method::POST)
                .uri(format!("{}/chat/completions", api_url))
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .header("Accept-Language", "en-US,en")
                .body(AsyncBody::from(request_body))
                .map_err(|error| LanguageModelCompletionError::from_cloud_failure(
                    PROVIDER_NAME,
                    "http_request_error".to_string(),
                    format!("Failed to build HTTP request: {}", error),
                    None,
                ))?;

            let mut response = http_client.send(http_request)
                .await
                .map_err(|error| LanguageModelCompletionError::from_cloud_failure(
                    PROVIDER_NAME,
                    "http_error".to_string(),
                    error.to_string(),
                    None,
                ))?;

            if !response.status().is_success() {
                return Err(LanguageModelCompletionError::from_cloud_failure(
                    PROVIDER_NAME,
                    "api_error".to_string(),
                    format!("API returned status: {}", response.status()),
                    None,
                ));
            }

            let mut body = String::new();
            response.body_mut().read_to_string(&mut body).await
                .map_err(|error| LanguageModelCompletionError::from_cloud_failure(
                    PROVIDER_NAME,
                    "response_read_error".to_string(),
                    format!("Failed to read response body: {}", error),
                    None,
                ))?;

            let mapper = ZAiEventMapper::new();
            Ok(mapper.map_response(body).boxed())
        });

        async move { Ok(future.await?.boxed()) }.boxed()
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
// Z.AI Response Structures
#[derive(Debug, Deserialize)]
pub struct ZAiResponse {
    pub id: String,
    pub request_id: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ZAiChoice>,
    pub usage: Option<ZAiUsage>,
}

#[derive(Debug, Deserialize)]
pub struct ZAiChoice {
    pub index: usize,
    pub message: ZAiMessageResponse,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ZAiMessageResponse {
    pub role: String,
    pub content: String,
    pub reasoning_content: Option<String>,
    pub tool_calls: Option<Vec<ZAiToolCallResponse>>,
}

#[derive(Debug, Deserialize)]
pub struct ZAiToolCallResponse {
    pub id: String,
    pub r#type: String,
    pub function: ZAiFunctionResponse,
}

#[derive(Debug, Deserialize)]
pub struct ZAiFunctionResponse {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct ZAiUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, EnumIter)]
pub enum ZAiModel {
    #[default]
    #[serde(rename = "glm-4.6")]
    Glm46,
    #[serde(rename = "glm-4.5")]
    Glm45,
    #[serde(rename = "glm-4.5-air")]
    Glm45Air,
    #[serde(rename = "custom")]
    Custom {
        name: String,
        display_name: Option<String>,
        max_tokens: u64,
        max_output_tokens: Option<u64>,
        supports_thinking: bool,
    },
}

impl ZAiModel {
    pub fn from_id(id: &str) -> Result<Self> {
        match id {
            "glm-4.6" => Ok(Self::Glm46),
            "glm-4.5" => Ok(Self::Glm45),
            "glm-4.5-air" => Ok(Self::Glm45Air),
            _ => anyhow::bail!("invalid model id: {id}"),
        }
    }

    pub fn id(&self) -> &str {
        match self {
            Self::Glm46 => "glm-4.6",
            Self::Glm45 => "glm-4.5",
            Self::Glm45Air => "glm-4.5-air",
            Self::Custom { name, .. } => name,
        }
    }

    pub fn display_name(&self) -> &str {
        match self {
            Self::Glm46 => "GLM-4.6",
            Self::Glm45 => "GLM-4.5",
            Self::Glm45Air => "GLM-4.5-Air",
            Self::Custom {
                name, display_name, ..
            } => display_name.as_ref().unwrap_or(name).as_str(),
        }
    }

    pub fn max_tokens(&self) -> u64 {
        match self {
            Self::Glm46 | Self::Glm45 | Self::Glm45Air => 128_000,
            Self::Custom { max_tokens, .. } => *max_tokens,
        }
    }

    pub fn max_output_tokens(&self) -> Option<u32> {
        match self {
            Self::Glm46 => Some(32_768),
            Self::Glm45 => Some(16_384),
            Self::Glm45Air => Some(8_192),
            Self::Custom {
                max_output_tokens, ..
            } => max_output_tokens.map(|t| t as u32),
        }
    }

    pub fn supports_thinking(&self) -> bool {
        match self {
            Self::Glm46 | Self::Glm45 | Self::Glm45Air => true,
            Self::Custom {
                supports_thinking, ..
            } => *supports_thinking,
        }
    }
}

pub use settings::ZAiAvailableModel;

pub struct ZAiEventMapper {
    tool_calls_by_index: std::collections::HashMap<usize, RawToolCall>,
}

#[derive(Default)]
struct RawToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl ZAiEventMapper {
    pub fn new() -> Self {
        Self {
            tool_calls_by_index: std::collections::HashMap::default(),
        }
    }

    // This method is no longer used - we use map_response instead
    #[allow(dead_code)]
    pub fn map_stream(
        self,
        _events: Pin<Box<dyn Send + futures::Stream<Item = Result<()>>>>,
    ) -> impl futures::Stream<Item = Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>
    {
        // Return an empty stream since this method is deprecated
        futures::stream::empty()
    }

    // This method is deprecated - we use process_response instead
    #[allow(dead_code)]
    pub fn map_event(
        &mut self,
        _event: (),
    ) -> Vec<Result<LanguageModelCompletionEvent, LanguageModelCompletionError>> {
        // This method is deprecated - we use process_response instead
        vec![]
    }

    fn process_response(
        &self,
        response_body: String,
    ) -> Vec<Result<LanguageModelCompletionEvent, LanguageModelCompletionError>> {
        let mut events = Vec::new();

        // Parse the Z.AI response
        let response: ZAiResponse = match serde_json::from_str(&response_body) {
            Ok(response) => response,
            Err(error) => {
                events.push(Err(LanguageModelCompletionError::from_cloud_failure(
                    PROVIDER_NAME,
                    "parse_error".to_string(),
                    format!("Failed to parse Z.AI response: {}", error),
                    None,
                )));
                return events;
            }
        };

        // Process each choice
        for choice in &response.choices {
            // Handle thinking content if present (Z.AI returns reasoning_content separate from main content)
            if let Some(reasoning_content) = &choice.message.reasoning_content {
                if !reasoning_content.is_empty() {
                    events.push(Ok(LanguageModelCompletionEvent::Thinking {
                        text: reasoning_content.clone(),
                        signature: None,
                    }));
                }
            }

            // Handle main content
            if !choice.message.content.is_empty() {
                events.push(Ok(LanguageModelCompletionEvent::Text(choice.message.content.clone())));
            }

            // Handle tool calls
            if let Some(tool_calls) = &choice.message.tool_calls {
                for tool_call in tool_calls {
                    let tool_use_id = LanguageModelToolUseId::from(tool_call.id.as_str());
                    let arguments = if tool_call.function.arguments.is_string() {
                        tool_call.function.arguments.as_str().unwrap_or("").to_string()
                    } else {
                        tool_call.function.arguments.to_string()
                    };

                    events.push(Ok(LanguageModelCompletionEvent::ToolUse(
                        LanguageModelToolUse {
                            id: tool_use_id.clone(),
                            name: tool_call.function.name.clone().into(),
                            raw_input: arguments.clone(),
                            input: serde_json::Value::String(arguments),
                            is_input_complete: true,
                        }
                    )));
                }
            }
        }

        // Send completion event
        if let Some(choice) = response.choices.first() {
            let stop_reason = match choice.finish_reason.as_deref() {
                Some("stop") => StopReason::EndTurn,
                Some("tool_calls") => StopReason::ToolUse,
                Some("length") => StopReason::MaxTokens,
                Some("sensitive") => StopReason::EndTurn, // Map sensitive content to end turn
                _ => StopReason::EndTurn,
            };

            events.push(Ok(LanguageModelCompletionEvent::Stop(stop_reason)));
        }

        // Send usage information if available
        if let Some(usage) = response.usage {
            events.push(Ok(LanguageModelCompletionEvent::UsageUpdate(TokenUsage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            })));
        }

        events
    }

    pub fn map_response(
        self,
        response_body: String,
    ) -> Pin<Box<dyn Send + futures::Stream<Item = Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>>> {
        // Use a simple implementation without async_stream
        Box::pin(futures::stream::iter(self.process_response(response_body)))
    }
}

pub struct ZAiConfigurationView {
    api_key_editor: Entity<InputField>,
    state: Entity<State>,
    load_credentials_task: Option<Task<()>>,
    target_agent: language_model::ConfigurationViewTargetAgent,
}

impl ZAiConfigurationView {
    fn new(
        state: Entity<State>,
        target_agent: language_model::ConfigurationViewTargetAgent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        cx.observe(&state, |_, _, cx| {
            cx.notify();
        })
        .detach();

        let load_credentials_task = Some(cx.spawn({
            let state = state.clone();
            async move |this, cx| {
                if let Some(task) = state
                    .update(cx, |state, cx| state.authenticate(cx))
                    .log_err()
                {
                    let _ = task.await;
                }
                this.update(cx, |this, cx| {
                    this.load_credentials_task = None;
                    cx.notify();
                })
                .log_err();
            }
        }));

        Self {
            api_key_editor: cx.new(|cx| InputField::new(window, cx, "Z.AI API Key")),
            state,
            load_credentials_task,
            target_agent,
        }
    }

    fn save_api_key(&mut self, _: &menu::Confirm, window: &mut Window, cx: &mut Context<Self>) {
        let api_key = self.api_key_editor.read(cx).text(cx);
        if api_key.is_empty() {
            return;
        }

        // Clear the editor after saving
        self.api_key_editor
            .update(cx, |editor, cx| editor.set_text("", window, cx));

        let state = self.state.clone();
        cx.spawn_in(window, async move |_, cx| {
            state
                .update(cx, |state, cx| state.set_api_key(Some(api_key), cx))?
                .await
        })
        .detach_and_log_err(cx);
    }
}

impl Focusable for ZAiConfigurationView {
    fn focus_handle(&self, cx: &App) -> FocusHandle {
        self.api_key_editor.read(cx).focus_handle(cx)
    }
}

impl Render for ZAiConfigurationView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let env_var_set = self.state.read(cx).api_key_state.is_from_env_var();

        if self.load_credentials_task.is_some() {
            div().child(Label::new("Loading credentials...")).into_any()
        } else if self.should_render_editor(cx) {
            v_flex()
                .size_full()
                .on_action(cx.listener(ZAiConfigurationView::save_api_key))
                .child(Label::new(format!("To use {}, you need to add an API key. Follow these steps:", match &self.target_agent {
                    language_model::ConfigurationViewTargetAgent::ZedAgent => "Zed's agent with Z.AI".into(),
                    language_model::ConfigurationViewTargetAgent::Other(agent) => agent.clone(),
                })))
                .child(
                    List::new()
                        .child(
                            crate::ui::InstructionListItem::new(
                                "Create one by visiting",
                                Some("Z.AI's settings"),
                                Some("https://z.ai/manage-apikey/apikey-list")
                            )
                        )
                        .child(
                            crate::ui::InstructionListItem::text_only("Paste your API key below and hit enter to start using the agent")
                        )
                )
                .child(self.api_key_editor.clone())
                .child(
                    Label::new(
                        format!("You can also assign the Z_AI_API_KEY environment variable and restart Zed."),
                    )
                    .size(LabelSize::Small)
                    .color(Color::Muted),
                )
                .into_any()
        } else {
            h_flex()
                .mt_1()
                .p_1()
                .justify_between()
                .rounded_md()
                .border_1()
                .border_color(cx.theme().colors().border_variant)
                .bg(cx.theme().colors().surface_background)
                .child(
                    h_flex()
                        .gap_2()
                        .items_center()
                        .child(Icon::new(IconName::Check))
                        .child(Label::new("API key configured"))
                )
                .when(!env_var_set, |this| {
                    this.child(
                        Button::new("remove-api-key", "Remove")
                            .size(ButtonSize::Compact)
                            .style(ButtonStyle::Filled)
                            .color(Color::Default)
                            .on_click(cx.listener(|this, _event, _window, cx| {
                                this.state.update(cx, |state, cx| {
                                    state.set_api_key(None, cx)
                                }).detach_and_log_err(cx);
                            }))
                    )
                })
                .into_any()
        }
    }
}

impl ZAiConfigurationView {
    fn should_render_editor(&self, cx: &App) -> bool {
        !self.state.read(cx).api_key_state.has_key() || self.state.read(cx).api_key_state.is_from_env_var()
    }
}