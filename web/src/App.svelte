<script lang="ts">
import { onMount } from "svelte"
import { API_AUTHORITY } from "./lib/api"
let messages = []

onMount(() => {
  let ws = new WebSocket(API_AUTHORITY + '/ws')
  ws.onmessage = (ev) => {
    messages = [...messages, String(ev.data)]
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
