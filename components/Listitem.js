import React from "react";
import { View, Text, TouchableOpacity, StyleSheet, Image } from "react-native";
const Listitem = ({name,symbol,currentPrice,priceChangePercentage7d,logoUrl}) => {
    const priceChangeColor = priceChangePercentage7d > 0 ? '#34C759':'#FF3B30'
  return (
    <TouchableOpacity>
      <View style={styles.ItemWrapper}>
        {/* Left side */}
        <View style={styles.leftWrapper}>
          <Image
            style={styles.image}
            source={{ uri:logoUrl} }
          />
          <View style={styles.titlesWrapper}>
            <Text style={styles.title}>{name}</Text>
            <Text style={styles.subtitle}>{symbol.toUpperCase()}</Text>
          </View>
        </View>
        {/* Right side */}
        <View style={styles.rightWrapper}>
          <Text style={styles.title}>${currentPrice.toLocaleString('en-US',{currency:'USD'})}</Text>
          <Text style={[styles.subtitle, { color: priceChangeColor }]}>{priceChangePercentage7d.toFixed(2)}%</Text>
        </View>
      </View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  ItemWrapper: {
    paddingHorizontal: 16,
    marginTop: 24,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  leftWrapper: {
    flexDirection: "row",
    alignItems: "center",
  },
  image: {
    height: 48,
    width: 48,
  },
  titlesWrapper: {
    marginLeft: 8,
  },
  title: {
    fontSize: 18,
  },
  subtitle: {
    marginTop: 4,
    fontSize: 14,
    color: "#A9ABB1",
  },
  rightWrapper: {
    alignItems: "flex-end",
  },
});
export default Listitem;
