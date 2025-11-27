# vLLM v1 KV Cache 운용 방식 분석

## 개요

vLLM v1은 완전히 재설계된 KV cache 관리 시스템을 가지고 있습니다. 이 문서는 v1의 KV cache가 어떻게 운용되는지 상세히 분석합니다.

## 아키텍처 개요

vLLM v1의 KV cache 관리 시스템은 다음과 같은 계층 구조로 구성되어 있습니다:

```
KVCacheManager (최상위 인터페이스)
    ↓
KVCacheCoordinator (KV cache 그룹 조정)
    ↓
SingleTypeKVCacheManager (단일 타입 KV cache 관리)
    ↓
BlockPool (블록 풀 관리)
```

## 주요 컴포넌트

### 1. KVCacheManager (`vllm/v1/core/kv_cache_manager.py`)

최상위 KV cache 관리자로, 스케줄러와 KV cache 시스템 사이의 인터페이스를 제공합니다.

#### 주요 메서드:

- **`get_computed_blocks(request)`**: 요청에 대해 이미 계산된(캐시된) 블록들을 찾아 반환
- **`allocate_slots(request, num_new_tokens, ...)`**: 새로운 토큰을 위한 KV cache 슬롯 할당
- **`free(request)`**: 요청이 완료되면 할당된 블록들을 해제
- **`cache_blocks(request, num_computed_tokens)`**: 계산된 블록들을 캐시에 저장

#### 동작 흐름:

1. **Prefix Cache Hit 확인**: `get_computed_blocks()`를 통해 요청의 프롬프트가 이미 캐시되어 있는지 확인
2. **블록 할당**: `allocate_slots()`를 통해 필요한 블록 수를 계산하고 할당
3. **블록 캐싱**: 계산이 완료된 블록들을 `cache_blocks()`로 캐시에 저장하여 재사용 가능하게 만듦
4. **블록 해제**: 요청 완료 시 `free()`로 블록을 해제

### 2. KVCacheCoordinator (`vllm/v1/core/kv_cache_coordinator.py`)

여러 KV cache 그룹을 조정하는 컴포넌트입니다. 모델이 여러 타입의 attention을 사용하는 경우(예: full attention + sliding window) 각 타입별로 별도의 KV cache 그룹을 관리합니다.

#### Coordinator 타입:

1. **KVCacheCoordinatorNoPrefixCache**: Prefix caching이 비활성화된 경우
2. **UnitaryKVCacheCoordinator**: 단일 KV cache 그룹만 있는 경우 (대부분의 모델)
3. **HybridKVCacheCoordinator**: 두 가지 타입의 KV cache 그룹이 있는 경우 (하나는 반드시 full attention)

#### 주요 기능:

- 여러 `SingleTypeKVCacheManager` 인스턴스를 조정
- 각 그룹별로 블록 할당/해제를 관리
- Prefix cache hit을 찾을 때 여러 그룹 간의 일관성 유지

### 3. SingleTypeKVCacheManager (`vllm/v1/core/single_type_kv_cache_manager.py`)

특정 타입의 attention에 대한 KV cache를 관리하는 추상 클래스입니다.

#### 구현체:

1. **FullAttentionManager**: Full attention 레이어용
2. **SlidingWindowManager**: Sliding window attention 레이어용
3. **ChunkedLocalAttentionManager**: Chunked local attention 레이어용
4. **MambaManager**: Mamba 모델용
5. **CrossAttentionManager**: Encoder-decoder 모델의 cross-attention용

#### 주요 메서드:

- **`allocate_new_blocks(request_id, num_tokens)`**: 새로운 블록 할당
- **`cache_blocks(request, num_tokens)`**: 블록을 캐시에 저장
- **`free(request_id)`**: 블록 해제
- **`find_longest_cache_hit(...)`**: 가장 긴 prefix cache hit 찾기
- **`remove_skipped_blocks(...)`**: 더 이상 필요 없는 블록 제거 (sliding window 등에서 사용)

### 4. BlockPool (`vllm/v1/core/block_pool.py`)

실제 KV cache 블록들을 관리하는 풀입니다.

#### 주요 구성 요소:

- **`blocks`**: 모든 KV cache 블록의 리스트
- **`free_block_queue`**: 사용 가능한 블록들의 큐 (LRU 순서)
- **`cached_block_hash_to_block`**: 블록 해시를 키로 하는 캐시 맵

#### 주요 메서드:

- **`get_new_blocks(num_blocks)`**: 새로운 블록 할당 (필요시 eviction 수행)
- **`free_blocks(ordered_blocks)`**: 블록들을 해제하여 풀로 반환
- **`cache_full_blocks(...)`**: 완전히 채워진 블록을 캐시에 저장
- **`get_cached_block(block_hash, kv_cache_group_ids)`**: 해시로 캐시된 블록 찾기
- **`touch(blocks)`**: 블록의 참조 카운트 증가 (prefix cache hit 시)

## KV Cache 블록 구조

### KVCacheBlock (`vllm/v1/core/kv_cache_utils.py`)

각 KV cache 블록은 다음 정보를 포함합니다:

```python
@dataclass
class KVCacheBlock:
    block_id: int              # 블록 ID (0 ~ num_gpu_blocks - 1)
    ref_cnt: int = 0           # 참조 카운트 (여러 요청이 공유할 수 있음)
    _block_hash: Optional[BlockHashWithGroupId] = None  # 블록 해시 (prefix caching용)
    prev_free_block: Optional[KVCacheBlock] = None     # 자유 블록 큐의 이전 블록
    next_free_block: Optional[KVCacheBlock] = None     # 자유 블록 큐의 다음 블록
    is_null: bool = False       # null 블록 여부
```

### BlockHash

블록의 내용을 해시하여 prefix caching에 사용합니다:

```python
class BlockHash(NamedTuple):
    hash_value: int                    # 블록의 해시 값
    token_ids: tuple[int, ...]         # 블록 내 토큰 ID들
    extra_keys: Optional[Any] = None  # 추가 키 (MM, LoRA 등)
```

## KV Cache 운용 흐름

### 1. 초기화 단계

1. **KV Cache Spec 생성**: 모델의 각 레이어에 대한 `KVCacheSpec` 생성
   - `FullAttentionSpec`, `SlidingWindowSpec`, `MambaSpec` 등
2. **KV Cache Config 생성**: `get_kv_cache_config()`를 통해 설정 생성
   - 사용 가능한 메모리 계산
   - 블록 수 결정
   - KV cache 그룹 구성
3. **BlockPool 초기화**: 모든 블록을 생성하고 free queue에 추가
4. **KVCacheManager 생성**: Coordinator와 함께 생성

### 2. 요청 스케줄링 단계

#### 2.1 새 요청 도착 시

```python
# 1. Prefix cache hit 확인
computed_blocks, num_computed_tokens = kv_cache_manager.get_computed_blocks(request)

# 2. 필요한 블록 수 계산 및 할당
new_blocks = kv_cache_manager.allocate_slots(
    request=request,
    num_new_tokens=num_new_tokens,
    num_new_computed_tokens=num_computed_tokens,
    new_computed_blocks=computed_blocks,
)
```

#### 2.2 Prefix Cache Hit 찾기

`get_computed_blocks()` 내부 동작:

1. 요청의 `block_hashes`를 순회하며 캐시에서 블록 찾기
2. 연속적으로 매칭되는 가장 긴 prefix 찾기
3. 매칭된 블록들의 참조 카운트 증가 (`touch()`)
4. 매칭된 블록들 반환

#### 2.3 블록 할당

`allocate_slots()` 내부 동작:

1. **스킵된 블록 제거**: Sliding window 등에서 더 이상 필요 없는 블록 제거
2. **필요한 블록 수 계산**: `get_num_blocks_to_allocate()` 호출
3. **블록 할당 가능 여부 확인**: free block 수 확인
4. **새 블록 할당**: `get_new_blocks()` 호출
   - 필요시 eviction 수행 (LRU 순서)
5. **블록 캐싱**: 계산 완료된 블록들을 캐시에 저장

### 3. 블록 할당 메커니즘

#### 3.1 블록 할당 (`BlockPool.get_new_blocks()`)

```python
def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
    # 1. Free queue에서 블록 가져오기
    ret = self.free_block_queue.popleft_n(num_blocks)
    
    # 2. 각 블록에 대해 eviction 처리
    for block in ret:
        if self.enable_caching:
            self._maybe_evict_cached_block(block)  # 캐시에서 제거
        block.ref_cnt += 1  # 참조 카운트 증가
    
    return ret
```

#### 3.2 Eviction 메커니즘

- **LRU (Least Recently Used)**: Free queue의 앞쪽 블록이 가장 먼저 evict됨
- **참조 카운트 기반**: `ref_cnt == 0`인 블록만 evict 가능
- **Eviction 시**: 블록의 해시를 제거하고 캐시 맵에서 삭제

### 4. 블록 캐싱

#### 4.1 캐싱 시점

블록이 완전히 채워진 후 (`num_full_blocks`만큼) 캐시에 저장됩니다:

```python
def cache_full_blocks(self, request, blocks, num_cached_blocks, num_full_blocks, ...):
    new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
    new_block_hashes = request.block_hashes[num_cached_blocks:]
    
    for i, blk in enumerate(new_full_blocks):
        block_hash = new_block_hashes[i]
        block_hash_with_group_id = BlockHashWithGroupId(block_hash, kv_cache_group_id)
        blk.block_hash = block_hash_with_group_id
        # 캐시 맵에 추가
        self.cached_block_hash_to_block[block_hash_with_group_id][blk.block_id] = blk
```

#### 4.2 블록 해시 생성

요청 생성 시 및 새 토큰 추가 시 블록 해시가 생성됩니다:

```python
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys):
    # 부모 블록 해시를 포함하여 체인 형태로 해시 생성
    return BlockHash(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys)),
        curr_block_token_ids_tuple,
        extra_keys
    )
```

### 5. 블록 해제

#### 5.1 요청 완료 시

```python
def free(self, request: Request) -> None:
    # 역순으로 블록 해제 (tail 블록이 먼저 evict되도록)
    self.coordinator.free(request.request_id)
```

#### 5.2 블록 해제 과정

```python
def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
    blocks_list = list(ordered_blocks)
    # 참조 카운트 감소
    for block in blocks_list:
        block.ref_cnt -= 1
    
    # 참조 카운트가 0이 된 블록들을 free queue에 추가
    self.free_block_queue.append_n([
        block for block in blocks_list
        if block.ref_cnt == 0 and not block.is_null
    ])
```

## 특수 케이스 처리

### 1. Sliding Window Attention

- **블록 제거**: 윈도우 밖의 블록은 `remove_skipped_blocks()`로 제거
- **Prefix Cache Hit**: 연속된 블록이 필요하므로 오른쪽에서 왼쪽으로 검색

### 2. Chunked Local Attention

- **블록 제거**: 현재 chunk 밖의 블록 제거
- **Prefix Cache Hit**: Local attention window 내의 블록만 검색

### 3. Hybrid Models (Full + Sliding Window)

- **HybridKVCacheCoordinator** 사용
- Full attention의 cache hit을 먼저 찾고, 그 범위 내에서 다른 타입의 cache hit 찾기
- 두 타입의 block_size가 배수 관계여야 함

### 4. Mamba Models

- 각 요청당 1개 블록만 할당
- Prefix caching 미지원

### 5. Cross-Attention (Encoder-Decoder)

- Encoder 상태는 요청별로 고유하므로 prefix caching 미지원
- Encoder 토큰 수에 따라 정적으로 할당

## 메모리 관리

### 블록 풀 크기 결정

```python
def get_num_blocks(vllm_config, num_layers, available_memory, page_size):
    # 사용 가능한 메모리를 레이어 수로 나눔
    num_blocks = int(available_memory // page_size // num_layers)
    return max(num_blocks, 0)
```

### 메모리 사용량 계산

각 KV cache spec은 `max_memory_usage_bytes()`를 구현하여 최대 메모리 사용량을 계산합니다:

- **FullAttentionSpec**: `max_model_len / block_size * page_size`
- **SlidingWindowSpec**: `(sliding_window - 1 + max_batched_tokens) / block_size * page_size`
- **ChunkedLocalAttentionSpec**: `(chunk_size + max_batched_tokens) / block_size * page_size`

## Prefix Caching 최적화

### 1. 블록 해시 체인

각 블록의 해시는 이전 블록의 해시를 포함하여 체인 형태로 구성됩니다. 이를 통해:
- 부분 블록 매칭 가능
- 해시 충돌 감소
- 순서 보장

### 2. 참조 카운트 관리

- 여러 요청이 같은 블록을 공유할 수 있음
- `ref_cnt`로 공유 블록 관리
- `ref_cnt == 0`일 때만 evict 가능

### 3. LRU Eviction

- Free queue는 LRU 순서로 유지됨
- Tail 블록부터 먼저 해제하여 최근 사용된 블록 보존

## 성능 최적화

### 1. Free Block Queue

- Doubly linked list로 구현하여 O(1) 삽입/삭제
- 중간 블록 제거도 O(1) 시간에 가능

### 2. 블록 해시 캐싱

- 요청 생성 시 모든 블록 해시를 미리 계산
- 새 토큰 추가 시에만 새 블록 해시 계산

### 3. 배치 처리

- 여러 블록을 한 번에 할당/해제하여 오버헤드 감소

## 주요 특징

1. **Paged Attention**: 블록 단위로 KV cache 관리
2. **Prefix Caching**: 동일한 프롬프트 prefix 재사용
3. **Hybrid Support**: 여러 attention 타입 동시 지원
4. **Memory Efficient**: Sliding window 등에서 불필요한 블록 제거
5. **Reference Counting**: 블록 공유를 통한 메모리 절약

## 요약

vLLM v1의 KV cache 시스템은 다음과 같은 특징을 가집니다:

- **계층적 구조**: Manager → Coordinator → SingleTypeManager → BlockPool
- **블록 기반 관리**: 고정 크기 블록으로 메모리 관리
- **Prefix Caching**: 블록 해시를 통한 자동 prefix 재사용
- **LRU Eviction**: 메모리 부족 시 최근 사용되지 않은 블록 제거
- **참조 카운트**: 여러 요청 간 블록 공유 지원
- **타입별 최적화**: 각 attention 타입에 맞는 최적화된 관리자

이러한 설계를 통해 vLLM v1은 효율적인 KV cache 관리와 높은 처리량을 달성합니다.
