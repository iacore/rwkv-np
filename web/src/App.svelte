<script lang="ts">
import { onMount } from "svelte"
import { API_AUTHORITY } from "./lib/api"
import { encode, decode } from "@msgpack/msgpack"

let messages = []

onMount(() => {
  let ws = new WebSocket(API_AUTHORITY + '/ws')
  ws.onmessage = async (ev) => {
    let data = await ev.data.arrayBuffer() 
    messages = [...messages, decode(data)]
  }
})
</script>

<main>
  <p>Hello world</p>
  <p>{API_AUTHORITY}</p>
  <ul>
    {#each messages as msg}
      <li>{msg}</li>
    {/each}
  </ul>
</main>

<style>
</style>
